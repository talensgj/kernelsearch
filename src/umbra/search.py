import logging
from typing import Optional
from functools import partial
from dataclasses import dataclass
from timeit import default_timer as timer
from importlib.metadata import version

import asdf
import numpy as np
from numpy.typing import ArrayLike
from scipy import signal
import multiprocessing as mp

from . import grid, utils, models, diagnostics


#############
#  LOGGING  #
#############

logger = logging.getLogger(__name__)

LOGDEBUG = logger.debug
LOGINFO = logger.info
LOGWARNING = logger.warning
LOGERROR = logger.error
LOGEXCEPTION = logger.exception


@dataclass
class PeriodGroup:
    """ Class for tracking properties of period groups.
    """

    period_idx: tuple[int, int]
    duration_idx: tuple[int, int]
    epoch_step: float
    search_mode: utils.SearchMode = None

    def get_period_group(self, period_grid):
        imin, imax = self.period_idx
        return period_grid[imin:imax]

    def get_duration_group(self, duration_grid):
        jmin, jmax = self.duration_idx
        return duration_grid[jmin:jmax]


def get_duration_idx(duration_grid: np.ndarray,
                     duration_lims: ArrayLike,
                     ) -> tuple[int, int]:
    """ Get the indices of a minimum an maximum duration into a duration grid.

    Parameters
    ----------
    duration_grid: np.ndarray
        An array of transit duration values.
    duration_lims: tuple[float, float]
        The minumum and maximum duration of interest.

    Returns
    -------
    jmin: int
        The index corresponding to the minimum duration.
    jmax: int
        The index corresponding to the maximum duration.

    """

    min_duration, max_duration = duration_lims

    jmin = np.searchsorted(duration_grid, min_duration, side='left')
    jmax = np.searchsorted(duration_grid, max_duration, side='right')

    return jmin, jmax


def make_period_groups(period_grid: np.ndarray,
                       duration_grid: np.ndarray,
                       duration_lims: grid.DurationLimits,
                       exp_time: float,
                       frac_duration_step: float = 1.05,
                       period_group_sampling: int = 3,
                       epoch_sampling: int = 20,
                       min_epoch_step: float = 1 / utils.MIN_IN_DAY,
                       max_epoch_step: float = 5 / utils.MIN_IN_DAY,
                       smooth_window: Optional[float] = None
                       ) -> list[PeriodGroup]:
    """ Split the full period range into groups to avoid cases
        where the max duration exceeds the min period.

    Parameters
    ----------
    period_grid: np.ndarray
        The array of period values to be searched in days.
    duration_grid: np.ndarray
        The array of duration values to be searched in days.
    duration_lims: DurationLimits
        The duration limits as a function of period in days.
    exp_time: float
        The exposure time of the observation in days. It is added to the
        transit durations to account for the smoothing effect of the
        integrations.
    frac_duration_step: float
        The ratio between consecutive durations in the grid. Equivalent to a
        grid with log-steps of log10(frac_duration_step) (default: 1.05).
    period_group_sampling: int
        The number of duration steps between the longest duration at the start
        of subsequent period groups.
    epoch_sampling: int
        The number of epoch steps to take in the shortest duration
        (default: 20).
    min_epoch_step: float
        The smallest acceptable epoch step in days (default: 1 minute).
    max_epoch_step: float
        The largest acceptable epoch step in days (default: 5 minutes).
    smooth_window: float, optional
        If given the second groups first period is choses so that it contains no
        cases where the smooth window contain multiple transits.

    Returns
    -------
    intervals: list[tuple[int, int]]
        The indeces needed to slice period_grid into the appropriate groups.

    """

    excess_duration_ratio = frac_duration_step ** period_group_sampling

    # Check that the duty cycle for all possible periods does not exceed the maximum value.
    duty_cycle = (excess_duration_ratio * duration_lims.long + exp_time)/period_grid
    if np.any(duty_cycle > utils.MAX_DUTY_CYCLE):
        msg = f"Longest duty cycle > {utils.MAX_DUTY_CYCLE:.2f}, reducing period_group_sampling is recommended."
        LOGWARNING(msg)

    # Identify the shortest period which is guaranteed to have only 1 transit in the smooth window.
    icut = -1
    if smooth_window is not None:

        # Guaranteed baseline if this period is the start of a period group.
        baseline = period_grid - excess_duration_ratio * duration_lims.long - exp_time

        # Index of shortest period with baseline > smooth_window.
        icut = np.searchsorted(baseline, smooth_window, side='right')

        if utils.DEBUG:
            diagnostics.plot_oot_baseline(period_grid, baseline, smooth_window)

    imin = 0
    intervals = []
    num_periods = len(period_grid)
    for imax in range(1, num_periods):

        if imax <= imin:
            continue

        # Create a period group break where WLS becomes fast.
        if imax == icut:
            LOGDEBUG(f"Adding split for smooth window.")
            intervals.append((imin, imax))
            imin = imax
            continue

        # Create period groups based on the ratio of max durations.
        if duration_lims.long[imax]/duration_lims.long[imin] > excess_duration_ratio:
            LOGDEBUG(f"Splitting periods on max duration ratio.")
            intervals.append((imin, imax))
            imin = imax
            continue

    # Add any remaining periods.
    if imin != num_periods:
        intervals.append((imin, num_periods))

    if utils.DEBUG:
        diagnostics.plot_period_groups(period_grid, duration_lims, intervals, icut)

    # Now that we know the period intervals, generate the auxillary data.
    ngroups = len(intervals)
    period_groups = []
    for group in range(ngroups):
        imin, imax = intervals[group]

        min_duration = np.amin(duration_lims.short[imin:imax])
        max_duration = np.amax(duration_lims.long[imin:imax])

        jmin, jmax = get_duration_idx(duration_grid, (min_duration, max_duration))

        epoch_step = grid.get_epoch_step(min_duration,
                                         epoch_sampling=epoch_sampling,
                                         min_epoch_step=min_epoch_step,
                                         max_epoch_step=max_epoch_step)

        group = PeriodGroup((imin, imax), (jmin, jmax), epoch_step)
        period_groups.append(group)

    return period_groups


def _search_period(period: np.ndarray,
                   duration_lims_circ: np.ndarray,
                   duration_lims_full: np.ndarray,
                   delta_time: np.ndarray,
                   weights_norm: np.ndarray,
                   delta_flux_weighted: np.ndarray,
                   flux_mean: float,
                   chisq0: float,
                   epoch_step: float,
                   duration_grid: np.ndarray,
                   templates: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                   min_points: np.ndarray,
                   is_short_period: bool,
                   normalisation: utils.Normalisation,
                   smooth_window: float,
                   smooth_weights: utils.SmoothWeights,
                   exp_time: float,
                   exp_cadence: float,
                   ld_type: utils.LDType,
                   ld_pars: ArrayLike,
                   debug: bool = False
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ Perform the transit search for a single period value.
    """

    # At short periods WLS requires special treatment.
    if is_short_period:
        templates = models.get_lstsq_templates(period,
                                               duration_grid,
                                               epoch_step,
                                               exp_time,
                                               exp_cadence,
                                               ld_type=ld_type,
                                               ld_pars=ld_pars,
                                               search_mode='WLS',
                                               smooth_window=smooth_window,
                                               smooth_weights=smooth_weights)

    # Unpack the transit templates.
    template_edges = templates[0]
    template_models = templates[1]
    template_square = templates[2]
    template_count = templates[3]

    # nrows: number of kernels (i.e. durations), ncols: length of transit kernels.
    nrows, ncols = template_models.shape

    # Phase fold the data.
    phase = np.mod(delta_time/period, 1)

    # Create the phase bins.
    num_bins = np.ceil(period/epoch_step).astype('int')
    bin_edges = np.linspace(0, 1, num_bins + 1)

    # Compute 'binned' quantities.
    # This is not a binning of the lightcurve, instead it is a nearest neighbour
    # interpolation into the template models.
    bin_idx = np.searchsorted(bin_edges, phase)
    count = np.bincount(bin_idx, minlength=num_bins + 2 + ncols - 1)
    a_bin = np.bincount(bin_idx, weights=weights_norm, minlength=num_bins + 2 + ncols - 1)
    b_bin = np.bincount(bin_idx, weights=delta_flux_weighted, minlength=num_bins + 2 + ncols - 1)

    # Remove out of bounds bins added by bincount.
    count = count[1:-1]
    a_bin = a_bin[1:-1]
    b_bin = b_bin[1:-1]

    # Extend arrays to allow for all epochs.
    count[num_bins:] = count[:ncols-1]
    a_bin[num_bins:] = a_bin[:ncols-1]
    b_bin[num_bins:] = b_bin[:ncols-1]
    bin_edges = np.append(bin_edges, bin_edges[1:ncols] + 1)

    # Reshape arrays to work with scipy.signal.oaconvolve.
    count = count.reshape((1, -1))
    a_bin = a_bin.reshape((1, -1))
    b_bin = b_bin.reshape((1, -1))

    # Perform convolutions.
    alpha = signal.oaconvolve(b_bin, template_models, mode='valid')
    beta = signal.oaconvolve(a_bin, template_square, mode='valid')
    gamma = signal.oaconvolve(a_bin, template_models, mode='valid')
    num_points = signal.oaconvolve(count, template_count, mode='valid')

    # Ignore division errors caused by an absence of in-transit data.
    # The invalid values are handled below.
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute transit depth scale factor.
        depth = alpha / (beta - gamma ** 2)

    # Handle epoch/duration combinations with few or no in-transit points.
    min_points = np.maximum(min_points, utils.IN_TRANSIT_FLOOR)

    if np.isscalar(min_points):
        mask = num_points < min_points
    else:
        mask = num_points < min_points[:, np.newaxis]

    depth[mask] = 0

    # Compute the delta chi-square.
    dchisq = alpha * depth

    # Split delta chi-square by flux increaes and flux decreases.
    # Use flux increases to establish a baseline.
    select_inc = depth < 0
    dchisq_inc = np.where(select_inc, dchisq, 0)
    dchisq_dec = np.where(select_inc, 0, dchisq)
    dchisq_inc = np.amax(dchisq_inc, axis=1)

    # Compute the power spectrum.
    if normalisation == 'simple':
        power = dchisq_dec/chisq0
    else:
        power = (dchisq_dec - dchisq_inc[:, np.newaxis])/(chisq0 - dchisq_inc[:, np.newaxis])

    if debug:
        phase_grid = (bin_edges[:-ncols] + bin_edges[ncols:]) / 2
        diagnostics.plot_power_at_period(phase_grid, duration_grid, power, depth, num_points, min_points)

    # For every duration find the peaks in the power, and associated dchisq values.
    irow = np.arange(power.shape[0])
    icol = np.argmax(power, axis=1)
    power = power[irow, icol]
    dchisq_dec = dchisq_dec[irow, icol]
    num_points = num_points[irow, icol]

    # Store the parameters corresponding to peak power values.
    phase_grid = (bin_edges[:-ncols] + bin_edges[ncols:])/2
    phase_vals = phase_grid[icol]
    depth_vals = depth[irow, icol]
    flux_level_vals = flux_mean - depth_vals * gamma[irow, icol]

    # Store the best model for the circular duration limits.
    jmin, jmax = get_duration_idx(duration_grid, duration_lims_circ)
    arg = np.argmax(power[jmin:jmax])
    best_power_circ = power[jmin + arg]
    best_model_circ = template_models[jmin + arg]
    best_vals_circ = (best_power_circ, template_edges, best_model_circ)

    # Store the best model for the full duration limits.
    jmin, jmax = get_duration_idx(duration_grid, duration_lims_full)
    arg = np.argmax(power[jmin:jmax])
    best_power_full = power[jmin + arg]
    best_model_full = template_models[jmin + arg]
    best_vals_full = (best_power_full, template_edges, best_model_full)

    return power, dchisq_dec, dchisq_inc, num_points, phase_vals, depth_vals, flux_level_vals, best_vals_circ, best_vals_full


def _search_periods(periods, duration_lims_circ, duration_lims_full, **kwargs):
    
    nrows = len(periods)
    ncols = len(kwargs['duration_grid'])
    power = np.full((nrows, ncols), np.nan)
    dchisq_dec = np.full((nrows, ncols), np.nan)
    dchisq_inc = np.full((nrows, ncols), np.nan)
    num_points = np.full((nrows, ncols), 0, dtype='uint32')
    phase_vals = np.full((nrows, ncols), np.nan)
    depth_vals = np.full((nrows, ncols), np.nan)
    flux_level_vals = np.full((nrows, ncols), np.nan)

    best_power_circ = -np.inf
    best_edges_circ = None
    best_model_circ = None

    best_power_full = -np.inf
    best_edges_full = None
    best_model_full = None

    search_func = partial(_search_period, **kwargs)
    for i, period in enumerate(periods):

        result = search_func(period, duration_lims_circ[i], duration_lims_full[i])

        power[i] = result[0]
        dchisq_dec[i] = result[1]
        dchisq_inc[i] = result[2]
        num_points[i] = result[3]
        phase_vals[i] = result[4]
        depth_vals[i] = result[5]
        flux_level_vals[i] = result[6]

        (power_circ, edges_circ, model_circ) = result[7]
        (power_full, edges_full, model_full) = result[8]

        if power_circ > best_power_circ:
            best_power_circ = power_circ
            best_edges_circ = edges_circ
            best_model_circ = model_circ

        if power_full > best_power_full:
            best_power_full = power_full
            best_edges_full = edges_full
            best_model_full = model_full

    best_vals_circ = (best_power_circ, best_edges_circ, best_model_circ)
    best_vals_full = (best_power_full, best_edges_full, best_model_full)

    return power, dchisq_dec, dchisq_inc, num_points, phase_vals, depth_vals, flux_level_vals, best_vals_circ, best_vals_full


def _search_periods_with_pool(num_processes, periods, duration_lims_circ, duration_lims_full, **kwargs):

    nrows = len(periods)
    ncols = len(kwargs['duration_grid'])
    power = np.full((nrows, ncols), np.nan)
    dchisq_dec = np.full((nrows, ncols), np.nan)
    dchisq_inc = np.full((nrows, ncols), np.nan)
    num_points = np.full((nrows, ncols), 0, dtype='uint32')
    phase_vals = np.full((nrows, ncols), np.nan)
    depth_vals = np.full((nrows, ncols), np.nan)
    flux_level_vals = np.full((nrows, ncols), np.nan)

    best_power_circ = -np.inf
    best_edges_circ = None
    best_model_circ = None

    best_power_full = -np.inf
    best_edges_full = None
    best_model_full = None

    search_func = partial(_search_periods, **kwargs)
    with mp.Pool(processes=num_processes) as pool:

        period_chunks = []
        for i in range(num_processes):

            periods_ = periods[i::num_processes]
            duration_lims_circ_ = duration_lims_circ[i::num_processes]
            duration_lims_full_ = duration_lims_full[i::num_processes]

            period_chunks.append((periods_, duration_lims_circ_, duration_lims_full_))

        i = 0
        for result in pool.starmap(search_func, period_chunks):

            power[i::num_processes, :] = result[0]
            dchisq_dec[i::num_processes, :] = result[1]
            dchisq_inc[i::num_processes, :] = result[2]
            num_points[i::num_processes, :] = result[3]
            phase_vals[i::num_processes, :] = result[4]
            depth_vals[i::num_processes, :] = result[5]
            flux_level_vals[i::num_processes, :] = result[6]

            (power_circ, edges_circ, model_circ) = result[7]
            (power_full, edges_full, model_full) = result[8]

            if power_circ > best_power_circ:
                best_power_circ = power_circ
                best_edges_circ = edges_circ
                best_model_circ = model_circ

            if power_full > best_power_full:
                best_power_full = power_full
                best_edges_full = edges_full
                best_model_full = model_full

            i += 1

    best_vals_circ = (best_power_circ, best_edges_circ, best_model_circ)
    best_vals_full = (best_power_full, best_edges_full, best_model_full)

    return power, dchisq_dec, dchisq_inc, num_points, phase_vals, depth_vals, flux_level_vals, best_vals_circ, best_vals_full


def _prepare_lightcurve(time: np.ndarray,
                        flux: np.ndarray,
                        flux_err: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """ Perform sanity checks and manipulate the input lightcurve.
    """

    # Make sure the input is sorted.
    sort = np.argsort(time)
    time = time[sort]
    flux = flux[sort]
    flux_err = flux_err[sort]

    # Compute time relative to the start of observations.
    tstart = time[0]
    delta_time = time - tstart

    # Compute normalized weights.
    weights = 1 / flux_err ** 2
    weights_sum = np.sum(weights)
    weights_norm = weights/weights_sum

    # Compute some quantities.
    flux_mean = np.sum(weights_norm * flux)
    delta_flux_weighted = weights_norm * (flux - flux_mean)
    chisq0 = np.sum(weights_norm * (flux - flux_mean) ** 2)

    return delta_time, delta_flux_weighted, weights_norm, tstart, flux_mean, weights_sum, chisq0


def _1d_periodogram(runtime: float,
                    period_grid: np.ndarray,
                    duration_grid: np.ndarray,
                    period_groups: list[PeriodGroup],
                    power: np.ndarray,
                    chisq0: float,
                    lc_size: int,
                    baseline: float,
                    dchisq_dec: np.ndarray,
                    dchisq_inc: np.ndarray,
                    num_points: np.ndarray,
                    midpoint_vals: np.ndarray,
                    depth_vals: np.ndarray,
                    flux_level_vals: np.ndarray,
                    best_template_vals: tuple,
                    duration_lims: grid.DurationLimits,
                    duration_circ: Optional[grid.DurationLimits] = None
                    ) -> tuple[dict, dict]:
    """ Collapse the 2D periodgram according the the given duration limits.
    """

    # Apply specific duration limits.
    mask = ((duration_grid[np.newaxis, :] >= duration_lims.short[:, np.newaxis])
            & (duration_grid[np.newaxis, :] <= duration_lims.long[:, np.newaxis]))

    power = np.where(mask, power, np.nan)

    # Temporary set all-NaNs rows are set to 0, so we can find the peak values.
    nanmask = np.all(np.isnan(power), axis=1, keepdims=True)
    power0 = np.where(nanmask, 0, power)

    # Find the peak values.
    irow = np.arange(power0.shape[0])
    icol = np.nanargmax(power0, axis=1)

    # Collapse the parameter arrays to 1D.
    power = power[irow, icol]
    dchisq_dec = dchisq_dec[irow, icol]
    dchisq_inc = dchisq_inc[irow, icol]
    num_points = num_points[irow, icol]
    midpoint_vals = midpoint_vals[irow, icol]
    duration_vals = duration_grid[icol]
    depth_vals = depth_vals[irow, icol]
    flux_level_vals = flux_level_vals[irow, icol]

    # Re-mask all-NaN rows.
    nanmask = np.squeeze(nanmask)
    power = np.where(nanmask, np.nan, power)
    dchisq_dec = np.where(nanmask, np.nan, dchisq_dec)
    dchisq_inc = np.where(nanmask, np.nan, dchisq_inc)
    num_points = np.where(nanmask, 0, num_points)
    midpoint_vals = np.where(nanmask, np.nan, midpoint_vals)
    duration_vals = np.where(nanmask, np.nan, duration_vals)
    depth_vals = np.where(nanmask, np.nan, depth_vals)
    flux_level_vals = np.where(nanmask, np.nan, flux_level_vals)

    # Compute the number of transits.
    phase_vals = np.mod(midpoint_vals/period_grid, 1)
    num_transits = np.zeros_like(period_grid)
    for i, (phase_, period_, duration_) in enumerate(zip(phase_vals, period_grid, duration_vals)):
        num_transits[i] = models.get_num_transits(phase_, period_, duration_, baseline)

    # Create status flags.
    status_flag = np.zeros_like(duration_vals, dtype='uint8')
    if duration_circ is not None:
        status_flag = np.where(duration_vals > duration_circ.long, 1, status_flag)
        status_flag = np.where(duration_vals < duration_circ.short, 2, status_flag)

    # Select the periodogram peak.
    ipeak = np.nanargmax(power)
    best_period = period_grid[ipeak]
    best_midpoint = midpoint_vals[ipeak]
    best_duration = duration_vals[ipeak]
    best_depth = depth_vals[ipeak]
    best_flux_level = flux_level_vals[ipeak]

    _, template_edges, template_model = best_template_vals
    phase_edges, model_flux = models.plot_lstsq_template(best_period,
                                                         best_depth,
                                                         best_flux_level,
                                                         template_edges,
                                                         template_model)

    # Save the periodogram header.
    header = dict()
    header['runtime'] = runtime
    header['chisq0'] = chisq0
    header['lc_size'] = lc_size
    header['baseline'] = baseline
    header['periods'] = period_grid
    header['durations'] = duration_grid

    groups = dict()
    groups['period_idx'] = [group.period_idx for group in period_groups]
    groups['duration_idx'] = [group.duration_idx for group in period_groups]
    groups['epoch_step'] = [group.epoch_step for group in period_groups]
    groups['templates'] = [group.search_mode for group in period_groups]
    header['groups'] = groups

    # Save the periodogram.
    periodogram = dict()
    periodogram['periods'] = period_grid
    periodogram['power'] = power
    periodogram['dchisq_dec'] = dchisq_dec
    periodogram['dchisq_inc'] = dchisq_inc
    periodogram['num_points'] = num_points
    periodogram['num_transits'] = num_transits
    periodogram['midpoint'] = midpoint_vals
    periodogram['duration'] = duration_vals
    periodogram['depth'] = depth_vals
    periodogram['flux_level'] = flux_level_vals
    periodogram['duration_short'] = duration_lims.short
    periodogram['duration_long'] = duration_lims.long
    if duration_circ is not None:
        periodogram['status_flag'] = status_flag

    # Save the best-fit model.
    parameters = dict()
    parameters['period'] = best_period
    parameters['midpoint'] = best_midpoint
    parameters['duration'] = best_duration
    parameters['depth'] = best_depth
    parameters['flux_level'] = best_flux_level

    transit_model = dict()
    transit_model['phase_edges'] = phase_edges
    transit_model['flux'] = model_flux
    transit_model['parameters'] = parameters

    # Create the search_result.
    search_result = dict()
    search_result['periodogram'] = periodogram
    search_result['transit_model'] = transit_model

    return header, search_result


def _transit_search(time: np.ndarray,
                    flux: np.ndarray,
                    flux_err: np.ndarray,
                    exp_time: float,
                    exp_cadence: float,
                    min_stellar_radius: float,
                    max_stellar_radius: float,
                    min_stellar_mass: float,
                    max_stellar_mass: float,
                    min_transits: int = 3,
                    min_separation: float = 3.,
                    period_sampling: int = 3,
                    min_period: Optional[float] = None,
                    max_period: Optional[float] = None,
                    epoch_sampling: int = 20,
                    min_epoch_step: float = 1 / utils.MIN_IN_DAY,
                    max_epoch_step: float = 5 / utils.MIN_IN_DAY,
                    circular_orbits: bool = True,
                    frac_eccentricity: float = 0.95,
                    frac_duration_step: float = 1.05,
                    period_group_sampling: int = 3,
                    normalisation: utils.Normalisation = 'umbra',
                    ld_type: utils.LDType = 'linear',
                    ld_pars: ArrayLike = (0.6,),
                    search_mode: utils.SearchMode = 'TLS',
                    short_periods: utils.ShortPeriods = 'skip',
                    smooth_window: Optional[float] = None,
                    smooth_weights: utils.SmoothWeights = 'uniform',
                    num_processes: Optional[int] = None,
                    ) -> tuple[dict, dict, dict]:
    """ Perform a transit search on the provided data.

    Parameters
    ----------
    time: np.ndarray
        The times of the observations in days.
    flux: np.ndarray
        The flux values of the observations.
    flux_err: np.ndarray
        The flux uncertainties of the observations.
    exp_time: float
        The exposure time of the observations in days.
    exp_cadence: float
        The exposure cadence of the observations in days.
    min_stellar_radius: float
        The lower bound on the stellar radius in solar units.
    max_stellar_radius: float
        The upper bound on the stellar radius in solar units.
    min_stellar_mass: float
        The lower bound on the stellar mass in solar units.
    max_stellar_mass: float
        The upper bound on the stellar mass in solar units.
    min_transits: int
        The minimum number of transits observed between the start and end of
        observations (default: 3).
    min_separation: float
        The minimum orbital separation between the host star and the planet in
        stellar radii (default: 3).
    period_sampling: int
        The oversampling factor of the period grid (default: 3).
    min_period: float
        The shortest period to search in days, overrides min_separation
        (default: None).
    max_period: float, optional
        The longest period to search in days, overrides min_transits
        (default: None).
    epoch_sampling: int
        The number of epoch steps to take in the shortest duration
        (default: 20).
    min_epoch_step: float
        The smallest acceptable epoch step in days (default: 1 minute).
    max_epoch_step: float
        The largest acceptable epoch step in days (default: 5 minutes).
    circular_orbits: bool
        If True, search transit durations appropriate for circular orbits,
        otherwise search a wider range for eccentric orbits (default: True).
    frac_eccentricity: float
        The fraction of the maximum stable eccentricity to consider, slightly
        limits the size of the duration space searched when circular_orbits =
        False (default: 0.95).
    frac_duration_step: float
        The ratio between consecutive durations in the grid. Equivalent to a
        grid with log-steps of log10(frac_duration_step) (default: 1.05).
    period_group_sampling: int
        The number of duration steps between the longest duration at the start
        of subsequent period groups.
    normalisation: str
        The way to convert the delta chi-square to periodogram power. Can be
        'normal' or 'umbra' (default: 'umbra').
    ld_type: str
        The limb-darkening law to use for the TLS or WLS templates. Can be any
        law valid in the batman package (default: 'linear').
    ld_pars: array-like
        The limb-darkening parameters to use (default: (0.6,)).
    search_mode: str
        The type of transit templates to use, can be 'BLS', 'TLS' or 'WLS'
        (default: 'TLS').
    short_periods: str
        How to treat short periods when search_mode = 'WLS', can be 'skip',
        'TLS' or 'WLS' (default: 'skip').
    smooth_window: float or None
        The smoothing window to use when search_mode = 'WLS', should match any
        whatever filter was applied to the data (default: None).
    smooth_weights: str
        The weights to apply across the smoothing window when search_mode = 'WLS',
        should match whatever filter was applied to the data and can be 'uniform'
        or 'tricube' (default: 'uniform').
    num_processes: int or None
        The number of CPUs to use when multi-processing (default: None).

    Returns
    -------
    header: dict
        The header of the periodogram search.
    search_result_circ: dict
        The periodogram and best-fit model for a search of the circular
        durations only.
    search_result_full: dict or None
        The periodogram and best-fit model for a search of the full (eccentric)
        duration range, provided only if circular_orbits = False.

    """

    # Check the input parameters.
    utils._verify_lightcurve(time, flux, flux_err)
    utils._verify_observation_params(exp_time, exp_cadence)
    utils._verify_stellar_params(min_stellar_radius, max_stellar_radius, min_stellar_mass, max_stellar_mass)
    utils._verify_period_grid_params(min_transits, min_separation, period_sampling, min_period, max_period)
    utils._verify_epoch_grid_params(epoch_sampling, min_epoch_step, max_epoch_step)
    utils._verify_duration_lims_params(circular_orbits, frac_eccentricity)
    utils._verify_frac_duration_step(frac_duration_step)
    utils._verify_period_group_sampling(period_group_sampling)
    utils._verify_normalisation(normalisation)
    ld_pars = utils._verify_ld_params(ld_type, ld_pars)
    smooth_window = utils._verify_lstsq_params(search_mode, smooth_window, smooth_weights, short_periods)
    num_processes = utils._verify_num_processes(num_processes)

    # Pre-compute some arrays from the lightcurves.
    result = _prepare_lightcurve(time, flux, flux_err)
    delta_time, delta_flux_weighted, weights_norm, tstart, flux_mean, weights_sum, chisq0 = result

    # Compute stellar densities from the stellar mass and radii ranges.
    min_stellar_density = utils.SOLAR_DENSITY * min_stellar_mass / max_stellar_radius ** 3
    max_stellar_density = utils.SOLAR_DENSITY * max_stellar_mass / min_stellar_radius ** 3
    stellar_density_bounds = (min_stellar_density, max_stellar_density)
    stellar_radius_bounds = (max_stellar_radius, min_stellar_radius)  # The max radius goes first because it matches the minimum density.

    # Compute the period grid to search.
    period_grid = grid.get_period_grid(max_stellar_density,
                                       delta_time[-1],
                                       min_period=min_period,
                                       max_period=max_period,
                                       oversampling=period_sampling,
                                       min_transits=min_transits,
                                       min_separation=min_separation)

    LOGINFO(f"Searching {len(period_grid)} periods between {period_grid[0]:.3f} days and {period_grid[-1]:.3f} days.")

    # Compute the duration limits as a function of orbital period.
    duration_circ, _, _ = grid.get_transit_duration_limits(period_grid,
                                                           stellar_density_bounds,
                                                           stellar_radius_bounds,
                                                           min_separation=min_separation,
                                                           circular_orbits=True)

    duration_full, _, _ = grid.get_transit_duration_limits(period_grid,
                                                           stellar_density_bounds,
                                                           stellar_radius_bounds,
                                                           min_separation=min_separation,
                                                           circular_orbits=False,
                                                           frac_eccentricity=frac_eccentricity)

    if circular_orbits:
        duration_lims = duration_circ
    else:
        duration_lims = duration_full

    # Compute the duration grid based on the duration limits.
    min_duration = np.amin(duration_lims.short)
    max_duration = np.amax(duration_lims.long)
    duration_grid = grid.get_transit_duration_grid(min_duration,
                                                   max_duration,
                                                   frac_duration_step=frac_duration_step)

    LOGINFO(f"Searching {len(duration_grid)} durations between {duration_grid[0]:.2f} days and {duration_grid[-1]:.2f} days.")

    # Compute the period groups.
    period_groups = make_period_groups(period_grid,
                                       duration_grid,
                                       duration_lims,
                                       exp_time,
                                       frac_duration_step=frac_duration_step,
                                       period_group_sampling=period_group_sampling,
                                       epoch_sampling=epoch_sampling,
                                       min_epoch_step=min_epoch_step,
                                       max_epoch_step=max_epoch_step,
                                       smooth_window=smooth_window)

    # Set up variables for the output.
    nrows = period_grid.size
    ncols = duration_grid.size

    # Initiate arrays to store the 2D periodogram results.
    power = np.full((nrows, ncols), fill_value=np.nan)
    dchisq_dec = np.full((nrows, ncols), fill_value=np.nan)
    dchisq_inc = np.full((nrows, ncols), fill_value=np.nan)
    num_points = np.full((nrows, ncols), fill_value=0, dtype='uint32')
    phase_vals = np.full((nrows, ncols), fill_value=np.nan)
    depth_vals = np.full((nrows, ncols), fill_value=np.nan)
    flux_level_vals = np.full((nrows, ncols), fill_value=np.nan)

    best_power_circ = -np.inf
    best_edges_circ = None
    best_model_circ = None

    best_power_full = -np.inf
    best_edges_full = None
    best_model_full = None

    runtime = 0
    ngroups = len(period_groups)
    for idx, group in enumerate(period_groups):
        start_time = timer()

        LOGDEBUG(f"Period group {idx + 1} of {ngroups}:")

        epoch_step = group.epoch_step
        imin, imax = group.period_idx
        jmin, jmax = group.duration_idx

        period_group = group.get_period_group(period_grid)
        duration_group = group.get_duration_group(duration_grid)

        duration_lims_circ = np.column_stack((duration_circ.short[imin:imax], duration_circ.long[imin:imax]))
        duration_lims_full = np.column_stack((duration_lims.short[imin:imax], duration_lims.long[imin:imax]))

        search_mode_ = search_mode
        is_short_period = False
        baseline = np.amin(period_group) - np.amax(duration_group) - exp_time
        if search_mode == 'WLS' and baseline < smooth_window:
            if short_periods == 'skip':
                LOGDEBUG("  Skipping short periods in WLS search.")
                continue
            if short_periods == 'TLS':
                search_mode_: utils.SearchMode = 'TLS'
                LOGDEBUG("  Using TLS templates for short periods in WLS search.")
            if short_periods == 'WLS':
                search_mode_: utils.SearchMode = 'TLS'
                is_short_period = True
                LOGDEBUG("  Using WLS templates for short periods in WLS search.")

        group.search_mode = search_mode_

        LOGDEBUG(f"  Searching {len(period_group)} periods between {period_group[0]:.3f} days to {period_group[-1]:.3f} days.")
        LOGDEBUG(f"  Searching {len(duration_group)} durations between {duration_group[0]:.2f} days and {duration_group[-1]:.2f} days.")
        LOGDEBUG(f"  Searching using an epoch step of {epoch_step * utils.SEC_IN_DAY / 60:.1f} minutes.")

        # Compute the template models for the current period set.
        templates = models.get_lstsq_templates(period_group,
                                               duration_group,
                                               epoch_step,
                                               exp_time,
                                               exp_cadence,
                                               ld_type=ld_type,
                                               ld_pars=ld_pars,
                                               search_mode=search_mode_,
                                               smooth_window=smooth_window,
                                               smooth_weights=smooth_weights)

        kwargs = dict()
        kwargs['delta_time'] = delta_time
        kwargs['weights_norm'] = weights_norm
        kwargs['delta_flux_weighted'] = delta_flux_weighted
        kwargs['flux_mean'] = flux_mean
        kwargs['chisq0'] = chisq0
        kwargs['epoch_step'] = epoch_step
        kwargs['duration_grid'] = duration_group
        kwargs['templates'] = templates
        kwargs['min_points'] = 0.5*duration_group/exp_cadence
        kwargs['is_short_period'] = is_short_period
        kwargs['normalisation'] = normalisation
        kwargs['smooth_window'] = smooth_window
        kwargs['smooth_weights'] = smooth_weights
        kwargs['exp_time'] = exp_time
        kwargs['exp_cadence'] = exp_cadence
        kwargs['ld_type'] = ld_type
        kwargs['ld_pars'] = ld_pars

        if num_processes is None:
            result = _search_periods(period_group, duration_lims_circ, duration_lims_full, **kwargs)
        else:
            result = _search_periods_with_pool(num_processes, period_group, duration_lims_circ, duration_lims_full, **kwargs)

        power[imin:imax, jmin:jmax] = result[0]
        dchisq_dec[imin:imax, jmin:jmax] = result[1]
        dchisq_inc[imin:imax, jmin:jmax] = result[2]
        num_points[imin:imax, jmin:jmax] = result[3]
        phase_vals[imin:imax, jmin:jmax] = result[4]
        depth_vals[imin:imax, jmin:jmax] = result[5]
        flux_level_vals[imin:imax, jmin:jmax] = result[6]

        (power_circ, edges_circ, model_circ) = result[7]
        (power_full, edges_full, model_full) = result[8]

        if power_circ > best_power_circ:
            best_power_circ = power_circ
            best_edges_circ = edges_circ
            best_model_circ = model_circ

        if power_full > best_power_full:
            best_power_full = power_full
            best_edges_full = edges_full
            best_model_full = model_full

        runtime_ = timer() - start_time
        runtime += runtime_
        LOGDEBUG(f"  Period group searched in {runtime_:.1f} seconds.")

    LOGINFO(f"Full search completed in {runtime:.1f} seconds.")

    best_vals_circ = (best_power_circ, best_edges_circ, best_model_circ)
    best_vals_full = (best_power_full, best_edges_full, best_model_full)

    # Multiply chi-square values with weights_sum.
    chisq0 *= weights_sum
    dchisq_dec *= weights_sum
    dchisq_inc *= weights_sum

    # Convert phase relative to start of observation to midpoint of first transit.
    midpoint_vals = tstart + period_grid[:, np.newaxis]*np.mod(phase_vals, 1)

    if utils.DEBUG:
        if circular_orbits:
            duration_full_ = None
        else:
            duration_full_ = duration_full

        diagnostics.plot_2d_periodogram(period_grid, duration_grid, power, dchisq_dec, dchisq_inc, num_points, midpoint_vals, depth_vals, flux_level_vals, duration_circ, duration_full_)

    # Generate the final periodogram for circular orbits.
    search_header, search_result_circ = _1d_periodogram(
        runtime,
        period_grid,
        duration_grid,
        period_groups,
        power,
        chisq0,
        delta_time.size,
        np.ptp(delta_time),
        dchisq_dec,
        dchisq_inc,
        num_points,
        midpoint_vals,
        depth_vals,
        flux_level_vals,
        best_vals_circ,
        duration_lims=duration_circ)

    if utils.DEBUG:
        periodogram = search_result_circ['periodogram']
        transit_model = search_result_circ['transit_model']

        diagnostics.plot_quick_look(time, flux, periodogram, transit_model, smooth_window)
        diagnostics.plot_1d_periodogram(periodogram)

    # Generate the final peridogram for the full duration range.
    search_result_full = None
    if not circular_orbits:
        search_header, search_result_full = _1d_periodogram(
            runtime,
            period_grid,
            duration_grid,
            period_groups,
            power,
            chisq0,
            delta_time.size,
            np.ptp(delta_time),
            dchisq_dec,
            dchisq_inc,
            num_points,
            midpoint_vals,
            depth_vals,
            flux_level_vals,
            best_vals_full,
            duration_lims=duration_full,
            duration_circ=duration_circ)

    if utils.DEBUG and search_result_full is not None:
        periodogram = search_result_full['periodogram']
        transit_model = search_result_full['transit_model']

        diagnostics.plot_quick_look(time, flux, periodogram, transit_model, smooth_window)
        diagnostics.plot_1d_periodogram(periodogram)

    return search_header, search_result_circ, search_result_full


class TransitSearch:

    def __init__(self,
                 min_transits: int = 3,
                 min_separation: float = 3.,
                 period_sampling: int = 3,
                 min_period: Optional[float] = None,
                 max_period: Optional[float] = None,
                 epoch_sampling: int = 20,
                 min_epoch_step: float = 1 / utils.MIN_IN_DAY,
                 max_epoch_step: float = 5 / utils.MIN_IN_DAY,
                 circular_orbits: bool = True,
                 frac_eccentricity: float = 0.95,
                 frac_duration_step: float = 1.05,
                 period_group_sampling: int = 3,
                 normalisation: utils.Normalisation = 'umbra',
                 num_processes: Optional[int] = None,
                 ):
        """ Initialise a transit search.
        """

        # Check the input parameters.
        utils._verify_period_grid_params(min_transits, min_separation, period_sampling, min_period, max_period)
        utils._verify_epoch_grid_params(epoch_sampling, min_epoch_step, max_epoch_step)
        utils._verify_duration_lims_params(circular_orbits, frac_eccentricity)
        utils._verify_frac_duration_step(frac_duration_step)
        utils._verify_period_group_sampling(period_group_sampling)
        utils._verify_normalisation(normalisation)
        num_processes = utils._verify_num_processes(num_processes)

        # Save the input parameters.
        self.min_transits = min_transits
        self.min_separation = min_separation
        self.period_sampling = period_sampling
        self.min_period = min_period
        self.max_period = max_period
        self.epoch_sampling = epoch_sampling
        self.min_epoch_step = min_epoch_step
        self.max_epoch_step = max_epoch_step
        self.circular_orbits = circular_orbits
        self.frac_eccentricity = frac_eccentricity
        self.frac_duration_step = frac_duration_step
        self.period_group_sampling = period_group_sampling
        self.normalisation = normalisation
        self.num_processes = num_processes

        return

    def _full_search_result(self,
                            target_config: dict,
                            search_header: dict,
                            search_result_circ: dict,
                            search_result_full: Optional[dict],
                            output_file: Optional[str]):

        # Build the global configuration section.
        global_config = dict()
        global_config['version'] = version('umbra')
        global_config['min_transits'] = self.min_transits
        global_config['min_separation'] = self.min_separation
        global_config['period_sampling'] = self.period_sampling
        global_config['min_period'] = self.min_period
        global_config['max_period'] = self.max_period
        global_config['epoch_sampling'] = self.epoch_sampling
        global_config['min_epoch_step'] = self.min_epoch_step
        global_config['max_epoch_step'] = self.max_epoch_step
        global_config['circular_orbits'] = self.circular_orbits
        global_config['frac_eccentricity'] = self.frac_eccentricity
        global_config['frac_duration_step'] = self.frac_duration_step
        global_config['period_group_sampling'] = self.period_group_sampling
        global_config['normalisation'] = self.normalisation
        global_config['num_processes'] = self.num_processes

        # Build the final file-tree.
        filetree = dict()
        filetree['config'] = {'global': global_config,
                              'target': target_config}
        filetree['search'] = {'header': search_header,
                              'circular': search_result_circ}
        if not self.circular_orbits:
            filetree['search']['eccentric'] = search_result_full

        # Convert the filtree to an asdf structure.
        full_result = asdf.AsdfFile(filetree)

        if output_file is not None:
            full_result.write_to(output_file)

        return full_result

    def boxy_lstsq(self,
                   time: np.ndarray,
                   flux: np.ndarray,
                   flux_err: np.ndarray,
                   exp_time: float,
                   exp_cadence: float,
                   min_stellar_radius: float,
                   max_stellar_radius: float,
                   min_stellar_mass: float,
                   max_stellar_mass: float,
                   output_file: Optional[str] = None
                   ):
        """ Perform a BLS-type transit search.
        """

        utils._verify_output_file(output_file)

        search_header, search_result_circ, search_result_full = _transit_search(
            time,
            flux,
            flux_err,
            exp_time,
            exp_cadence,
            min_stellar_radius,
            max_stellar_radius,
            min_stellar_mass,
            max_stellar_mass,
            min_transits=self.min_transits,
            min_separation=self.min_separation,
            period_sampling=self.period_sampling,
            min_period=self.min_period,
            max_period=self.max_period,
            epoch_sampling=self.epoch_sampling,
            min_epoch_step=self.min_epoch_step,
            max_epoch_step=self.max_epoch_step,
            circular_orbits=self.circular_orbits,
            frac_eccentricity=self.frac_eccentricity,
            frac_duration_step=self.frac_duration_step,
            period_group_sampling=self.period_group_sampling,
            normalisation=self.normalisation,
            search_mode="BLS",
            num_processes=self.num_processes)

        target_config = dict()
        target_config['templates'] = 'BLS'
        target_config['exp_time'] = exp_time
        target_config['exp_cadence'] = exp_cadence
        target_config['min_stellar_radius'] = min_stellar_radius
        target_config['max_stellar_radius'] = max_stellar_radius
        target_config['min_stellar_mass'] = min_stellar_mass
        target_config['max_stellar_mass'] = max_stellar_mass

        full_result = self._full_search_result(target_config,
                                               search_header,
                                               search_result_circ,
                                               search_result_full,
                                               output_file)

        return full_result

    def transit_lstsq(self,
                      time: np.ndarray,
                      flux: np.ndarray,
                      flux_err: np.ndarray,
                      exp_time: float,
                      exp_cadence: float,
                      min_stellar_radius: float,
                      max_stellar_radius: float,
                      min_stellar_mass: float,
                      max_stellar_mass: float,
                      ld_type: utils.LDType,
                      ld_pars: ArrayLike,
                      output_file: Optional[str] = None
                      ):
        """ Perform a TLS-type transit search.
        """

        utils._verify_output_file(output_file)

        search_header, search_result_circ, search_result_full = _transit_search(
            time,
            flux,
            flux_err,
            exp_time,
            exp_cadence,
            min_stellar_radius,
            max_stellar_radius,
            min_stellar_mass,
            max_stellar_mass,
            min_transits=self.min_transits,
            min_separation=self.min_separation,
            period_sampling=self.period_sampling,
            min_period=self.min_period,
            max_period=self.max_period,
            epoch_sampling=self.epoch_sampling,
            min_epoch_step=self.min_epoch_step,
            max_epoch_step=self.max_epoch_step,
            circular_orbits=self.circular_orbits,
            frac_eccentricity=self.frac_eccentricity,
            frac_duration_step=self.frac_duration_step,
            period_group_sampling=self.period_group_sampling,
            normalisation=self.normalisation,
            ld_type=ld_type,
            ld_pars=ld_pars,
            search_mode="TLS",
            num_processes=self.num_processes)

        target_config = dict()
        target_config['templates'] = 'TLS'
        target_config['exp_time'] = exp_time
        target_config['exp_cadence'] = exp_cadence
        target_config['min_stellar_radius'] = min_stellar_radius
        target_config['max_stellar_radius'] = max_stellar_radius
        target_config['min_stellar_mass'] = min_stellar_mass
        target_config['max_stellar_mass'] = max_stellar_mass
        target_config['ld_type'] = ld_type
        target_config['ld_pars'] = np.asarray(ld_pars)

        full_result = self._full_search_result(target_config,
                                               search_header,
                                               search_result_circ,
                                               search_result_full,
                                               output_file)

        return full_result

    def warped_lstsq(self,
                     time: np.ndarray,
                     flux: np.ndarray,
                     flux_err: np.ndarray,
                     exp_time: float,
                     exp_cadence: float,
                     min_stellar_radius: float,
                     max_stellar_radius: float,
                     min_stellar_mass: float,
                     max_stellar_mass: float,
                     ld_type: utils.LDType,
                     ld_pars: ArrayLike,
                     smooth_window: float,
                     smooth_weights: utils.SmoothWeights = 'uniform',
                     short_periods: utils.ShortPeriods = 'skip',
                     output_file: Optional[str] = None
                     ):
        """ Perform a WLS-type transit search.
        """

        utils._verify_output_file(output_file)

        search_header, search_result_circ, search_result_full = _transit_search(
            time,
            flux,
            flux_err,
            exp_time,
            exp_cadence,
            min_stellar_radius,
            max_stellar_radius,
            min_stellar_mass,
            max_stellar_mass,
            min_transits=self.min_transits,
            min_separation=self.min_separation,
            period_sampling=self.period_sampling,
            min_period=self.min_period,
            max_period=self.max_period,
            epoch_sampling=self.epoch_sampling,
            min_epoch_step=self.min_epoch_step,
            max_epoch_step=self.max_epoch_step,
            circular_orbits=self.circular_orbits,
            frac_eccentricity=self.frac_eccentricity,
            frac_duration_step=self.frac_duration_step,
            period_group_sampling=self.period_group_sampling,
            normalisation=self.normalisation,
            ld_type=ld_type,
            ld_pars=ld_pars,
            search_mode="WLS",
            short_periods=short_periods,
            smooth_window=smooth_window,
            smooth_weights=smooth_weights,
            num_processes=self.num_processes)

        target_config = dict()
        target_config['templates'] = 'WLS'
        target_config['exp_time'] = exp_time
        target_config['exp_cadence'] = exp_cadence
        target_config['min_stellar_radius'] = min_stellar_radius
        target_config['max_stellar_radius'] = max_stellar_radius
        target_config['min_stellar_mass'] = min_stellar_mass
        target_config['max_stellar_mass'] = max_stellar_mass
        target_config['ld_type'] = ld_type
        target_config['ld_pars'] = np.asarray(ld_pars)
        target_config['smooth_window'] = smooth_window
        target_config['smooth_weights'] = smooth_weights
        target_config['short_periods'] = short_periods

        full_result = self._full_search_result(target_config,
                                               search_header,
                                               search_result_circ,
                                               search_result_full,
                                               output_file)

        return full_result


def main():
    return


if __name__ == '__main__':
    main()
