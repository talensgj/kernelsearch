import logging
from typing import Optional
from functools import partial
from collections import namedtuple

import numpy as np
from scipy import signal
import multiprocessing as mp

from astropy import constants

from . import grid, models, diagnostics
# from transitleastsquares import grid, tls_constants

import matplotlib.pyplot as plt


#############
#  LOGGING  #
#############

logger = logging.getLogger(__name__)

LOGDEBUG = logger.debug
LOGINFO = logger.info
LOGWARNING = logger.warning
LOGERROR = logger.error
LOGEXCEPTION = logger.exception


###########
# GLOBALS #
###########

SECINDAY = 24*3600
SOLAR_DENSITY = (constants.M_sun/(4/3 * np.pi * constants.R_sun ** 3)).to('g/cm^3').value


def evaluate_template(time,
                      period,
                      midpoint,
                      depth,
                      flux_level,
                      template_edges,
                      template_model):

    phase = np.mod((time - midpoint) / period - 0.5, 1)  # Phase with transit at 0.5
    bin_idx = np.searchsorted(template_edges / period + 0.5, phase)  # Phase centered at 0.5
    template_model = np.append(np.append(0, template_model), 0)
    flux = depth*template_model[bin_idx] + flux_level

    return phase, flux


# def _get_duration_lims(period,
#                        R_star_min,
#                        R_star_max,
#                        M_star_min,
#                        M_star_max):
#
#     max_duration = grid.T14(
#         R_s=R_star_max,
#         M_s=M_star_max,
#         P=period,
#         small=False  # large planet for long transit duration
#     )
#
#     min_duration = grid.T14(
#         R_s=R_star_min,
#         M_s=M_star_min,
#         P=period,
#         small=True  # small planet for short transit duration
#     )
#
#     min_duration = min_duration*period
#     max_duration = max_duration*period
#
#     return min_duration, max_duration


# def get_duration_lims(periods,
#                       R_star_min,
#                       R_star_max,
#                       M_star_min,
#                       M_star_max):
#
#     min_duration0, max_duration0 = _get_duration_lims(np.amin(periods), R_star_min, R_star_max, M_star_min, M_star_max)
#     min_duration1, max_duration1 = _get_duration_lims(np.amax(periods), R_star_min, R_star_max, M_star_min, M_star_max)
#
#     min_duration = np.minimum(min_duration0, min_duration1)
#     max_duration = np.maximum(max_duration0, max_duration1)
#
#     return min_duration, max_duration


# def _get_epoch_step(min_duration,
#                   ref_period,
#                   ref_depth: float = 0.005,
#                   exp_time: Optional[float] = None,
#                   min_epoch_step: float = 1 / (24 * 60),  # TODO are these good values?
#                   max_epoch_step: float = 5 / (24 * 60),  # TODO are these good values?
#                   oversampling: int = 3):
#
#     if exp_time is None:
#         exp_time = 0.
#
#     # Compute the approriate epoch step.
#     axis = models.duration2axis(min_duration,
#                                 ref_period,
#                                 np.sqrt(ref_depth),
#                                 0., 0., 90.)
#     full = models.axis2full(axis,
#                             ref_period,
#                             np.sqrt(ref_depth),
#                             0., 0., 90.)
#
#     ingress_time = (min_duration - full) / 2
#     ingress_time = np.maximum(ingress_time, exp_time)
#     epoch_step = ingress_time / oversampling
#
#     if epoch_step < min_epoch_step:
#         epoch_step = min_epoch_step
#
#     if epoch_step > max_epoch_step:
#         epoch_step = max_epoch_step
#
#     return epoch_step


# def _duration_grid(min_duration: float,
#                    max_duration: float,
#                    ref_period: float,
#                    ref_depth: float = 0.005,
#                    oversampling: float = 4):
#
#     duration = min_duration
#     duration_grid = [min_duration]
#     while duration < max_duration:
#         axis = models.duration2axis(duration,
#                                     ref_period,
#                                     np.sqrt(ref_depth),
#                                     0., 0., 90.)
#         full = models.axis2full(axis,
#                                 ref_period,
#                                 np.sqrt(ref_depth),
#                                 0., 0., 90.)
#
#         # Increment by fractions of the previous transits ingress/egress.
#         duration_step = (duration - full) / oversampling
#         duration = duration + duration_step
#
#         if duration < max_duration:
#             duration_grid.append(duration)
#         else:
#             duration_grid.append(max_duration)
#
#     duration_grid = np.array(duration_grid)
#
#     return duration_grid


# def get_duration_grid(periods: np.ndarray,
#                       R_star_min: float,
#                       R_star_max: float,
#                       M_star_min: float,
#                       M_star_max: float,
#                       ref_depth: float = 0.005,
#                       exp_time: Optional[float] = None,
#                       min_epoch_step: float = 1/(24*60),  # TODO are these good values?
#                       max_epoch_step: float = 5/(24*60),  # TODO are these good values?
#                       oversampling_epoch: int = 3,
#                       oversampling_duration: float = 4):
#
#     ref_period = np.amax(periods)
#     min_duration, max_duration = get_duration_lims(periods, R_star_min, R_star_max, M_star_min, M_star_max)
#
#     epoch_step = _get_epoch_step(min_duration,
#                              ref_period,
#                              ref_depth=ref_depth,
#                              exp_time=exp_time,
#                              min_epoch_step=min_epoch_step,
#                              max_epoch_step=max_epoch_step,
#                              oversampling=oversampling_epoch)
#
#     duration_grid = _duration_grid(min_duration,
#                                    max_duration,
#                                    ref_period,
#                                    ref_depth=ref_depth,
#                                    oversampling=oversampling_duration)
#
#     return epoch_step, duration_grid


# TODO This function needs to be reconsidered. The grouping is necessary to
# TODO speed up computation time, but I'm not sure the current implementation
# TODO is optimal.
def make_period_groups(period_grid: np.ndarray,
                       exp_time: float,
                       duration_lims: grid.DurationLimits,
                       max_duty_cycle: float = 0.20,
                       smooth_window: Optional[float] = None
                       ) -> list[tuple[int, int]]:
    """ Split the full period range into groups to avoid cases
        where the max duration exceeds the min period.

    Parameters
    period_grid: np.ndarray
        The array of period values to be searched in days.
    exp_time: float
        The exposure time of the observation in days. It is added to the
        transit durations to account for the smoothing effect of the
        integrations.
    max_duty_cycle: float
        The maximum ratio of the max duration (+ exp_time) over the min period
        within a period group.
    smooth_window: float, optional
        If given the second groups first period is choses so that it contains no
        cases where the smooth window contain multiple transits.

    Returns
    -------
    intervals: list[tuple[int, int]]
        The indeces needed to slice period_grid into the appropriate groups.

    """

    imin = 0
    intervals = []
    if smooth_window is not None:
        # Guaranteed baseline if this period is the start of a period group.
        baseline = (1 - max_duty_cycle)*period_grid

        # Index of shortest period with baseline > smooth_window
        imin = np.searchsorted(baseline, smooth_window, side='right')

        if imin > 0:
            intervals = [(0, imin)]

    for i, period in enumerate(period_grid):
        if i < imin:
            continue

        max_duration = duration_lims.long[i]
        if (max_duration + exp_time)/period_grid[imin] > max_duty_cycle:
            intervals.append((imin, i))
            imin = i

    if imin != len(period_grid):
        intervals.append((imin, len(period_grid)))

    return intervals
        
        
def _make_transit_templates(mid_times: np.ndarray,
                            duration_grid: np.ndarray,
                            transit_params: dict,
                            supersample_factor: int,
                            ld_type: str,
                            ld_pars: tuple,
                            exp_time: float,
                            exp_cadence: float,
                            search_mode: str = 'TLS',
                            smooth_window: Optional[float] = None,
                            smooth_weights: str = 'uniform'):

    if search_mode == 'WLS':

        # Create the grid of exposures inside the smoothing window.
        nevals = np.ceil(smooth_window / exp_cadence).astype('int')
        if nevals % 2 == 0:
            nevals += 1

        mid_idx = nevals // 2
        dt = (np.arange(nevals) - mid_idx) * exp_cadence

        # Compute the weights across the smoothing window.
        if smooth_weights == 'uniform':
            weights = np.ones_like(dt)

        if smooth_weights == 'tricube':
            radius = smooth_window/2
            weights = np.where(np.abs(dt) < radius, (1 - np.abs(dt/radius)**3)**3, 0)

        # Normalize the weights.
        weights = weights/np.sum(weights)

        # Make the dt and weights values 2D.
        dt = dt[:, np.newaxis]
        weights = weights[:, np.newaxis]

        # Get the full array of transit times needed to compute warped transits.
        dt = dt + mid_times[np.newaxis, :]
        dt_shape = dt.shape
        dt = dt.ravel()

    nrows = len(duration_grid)
    ncols = len(mid_times)
    bls_template = np.zeros((nrows, ncols))
    tls_template = np.zeros((nrows, ncols))
    wls_template = np.zeros((nrows, ncols))
    for row_idx, duration in enumerate(duration_grid):

        # Compute the scaled semi-major axis that gives the required duration.
        axis = models.duration2axis(duration,
                                    transit_params['P'],
                                    transit_params['R_p/R_s'],
                                    transit_params['b'],
                                    transit_params['ecc'],
                                    transit_params['w'])
        transit_params['a/R_s'] = axis

        # Evaluate the transit model.
        result = models.analytic_transit_model(mid_times,
                                               transit_params,
                                               'uniform',
                                               [],
                                               exp_time=exp_time,
                                               supersample_factor=supersample_factor,
                                               max_err=1.)
        bls_template[row_idx] = result[0]

        result = models.analytic_transit_model(mid_times,
                                               transit_params,
                                               ld_type,
                                               ld_pars,
                                               exp_time=exp_time,
                                               supersample_factor=supersample_factor,
                                               max_err=1.)
        fac = result[5]
        tls_template[row_idx] = result[0]

        if search_mode == 'WLS':

            # Evaluate the transit model.
            result = models.analytic_transit_model(dt,
                                                   transit_params,
                                                   ld_type,
                                                   ld_pars,
                                                   exp_time=exp_time,
                                                   supersample_factor=supersample_factor,
                                                   fac=fac,
                                                   max_err=1.)

            flux_dt = result[0]
            flux_dt = flux_dt.reshape(dt_shape)
            wls_template[row_idx] = flux_dt[mid_idx]/np.sum(weights*flux_dt, axis=0)
            
    return bls_template, tls_template, wls_template


def make_template_grid(periods: np.ndarray,
                       duration_grid: np.ndarray,
                       epoch_step: float,
                       exp_time: float,
                       exp_cadence: float,
                       ld_type: str = 'linear',
                       ld_pars: tuple = (0.6,),
                       ref_depth: float = 0.005,
                       search_mode: str = 'TLS',
                       smooth_window: Optional[float] = None,
                       smooth_weights: str = 'uniform'
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if search_mode not in ['BLS', 'TLS', 'WLS']:
        errmsg = f"Invalid value '{search_mode}' for parameter search_mode."
        raise ValueError(errmsg)

    if search_mode == 'WLS' and smooth_window is None:
        errmsg = f"Parameter smooth_window can not be None for WLS search."
        raise ValueError(errmsg)

    if smooth_weights not in ['uniform', 'tricube']:
        errmsg = f"Invalid value '{smooth_weights}' for parameter smooth_weights."
        raise ValueError(errmsg)

    min_period = np.amin(periods)
    max_period = np.amax(periods)
    max_duration = np.amax(duration_grid)

    baseline = min_period - max_duration - exp_time
    if search_mode == 'WLS' and periods.size > 1 and baseline < smooth_window:
        LOGWARNING("Cannot make WLS templates for this period range, defaulting to TLS templates.")
        search_mode = 'TLS'

    if search_mode in ['BLS', 'TLS']:
        delta_time = max_duration + exp_time
    else:
        delta_time = max_duration + exp_time + smooth_window

    if periods.size == 1:
        delta_time = np.minimum(delta_time, max_period)

    # Determine the times at which to evaluate the template.
    nbins = np.ceil(delta_time/epoch_step).astype('int')
    template_edges = np.linspace(-delta_time/2, delta_time/2, nbins + 1)
    mid_times = (template_edges[:-1] + template_edges[1:])/2

    # Set up the transit parameters.
    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = max_period
    transit_params['R_p/R_s'] = np.sqrt(ref_depth)
    transit_params['a/R_s'] = 0.
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    supersample_factor = np.ceil(exp_time * SECINDAY / 10.).astype('int')

    # Compute the transit templates.
    result = _make_transit_templates(mid_times,
                                     duration_grid,
                                     transit_params,
                                     supersample_factor,
                                     ld_type,
                                     ld_pars,
                                     exp_time,
                                     exp_cadence,
                                     search_mode=search_mode,
                                     smooth_window=smooth_window,
                                     smooth_weights=smooth_weights)
    bls_template, tls_template, wls_template = result

    # Choose the final template based on the search mode.
    if search_mode == 'BLS':
        template_models = (bls_template - 1)/ref_depth
    if search_mode == 'TLS':
        template_models = (tls_template - 1)/ref_depth
    if search_mode == 'WLS':
        template_models = (wls_template - 1)/ref_depth

    template_square = template_models ** 2
    template_count = (bls_template - 1) < 0

    return template_edges, template_models, template_square, template_count


def _search_period(period,
                   min_duration,
                   max_duration,
                   time,
                   weights_norm,
                   delta_flux_weighted,
                   flux_mean,
                   chisq0,
                   epoch_step,
                   duration_grid,
                   templates,
                   min_points,
                   is_short_period,
                   normalisation,
                   smooth_window,
                   smooth_weights,
                   exp_time,
                   exp_cadence,
                   ld_type,
                   ld_pars,
                   debug=False
                   ):

    nvals = duration_grid.size

    # Select the durations that encompass the required range at this period.
    jmin = np.searchsorted(duration_grid, min_duration, side='left')
    jmax = np.searchsorted(duration_grid, max_duration, side='right')

    min_points = min_points[jmin:jmax]
    duration_grid = duration_grid[jmin:jmax]

    template_edges = templates[0]
    template_models = templates[1][jmin:jmax]
    template_square = templates[2][jmin:jmax]
    template_count = templates[3][jmin:jmax]

    # At short periods WLS requires special treatment.
    if is_short_period:
        templates = make_template_grid(period,
                                       duration_grid,
                                       epoch_step,
                                       exp_time,
                                       exp_cadence,
                                       ld_type=ld_type,
                                       ld_pars=ld_pars,
                                       search_mode='WLS',
                                       smooth_window=smooth_window,
                                       smooth_weights=smooth_weights)
        template_edges = templates[0]
        template_models = templates[1]
        template_square = templates[2]
        template_count = templates[3]

    # nrows: number of kernels (i.e. durations), ncols: length of transit kernels.
    nrows, ncols = template_models.shape

    # Phase fold the data.
    phase = np.mod(time/period, 1)

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
    npoints = signal.oaconvolve(count, template_count, mode='valid')
    alpha = signal.oaconvolve(b_bin, template_models, mode='valid')
    beta = signal.oaconvolve(a_bin, template_square, mode='valid')
    gamma = signal.oaconvolve(a_bin, template_models, mode='valid')

    # Ignore division errors caused by an absence of in-transit data.
    # The invalid values are handled below.
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute transit depth scale factor.
        depth = alpha / (beta - gamma ** 2)

    # Handle epoch/duration combinations with few or no in-transit points.
    # All elements of min_points must be >=1.
    min_points = np.maximum(min_points, 1)

    if np.isscalar(min_points):
        mask = npoints < min_points
    else:
        mask = npoints < min_points[:, np.newaxis]

    depth[mask] = 0

    # Compute the delta chi-square.
    dchisq = alpha * depth

    # Split delta chi-square by flux increaes and flux decreases.
    # Use flux increases to establish a baseline.
    select_inc = depth < 0
    dchisq_inc = np.where(select_inc, dchisq, 0)
    dchisq_dec = np.where(select_inc, 0, dchisq)
    dchisq_inc_ = np.amax(dchisq_inc, axis=1)

    # Compute the power spectrum.
    if normalisation == 'normal':
        power = dchisq_dec/chisq0
    else:
        power = (dchisq_dec - dchisq_inc_[:, np.newaxis])/(chisq0 - dchisq_inc_[:, np.newaxis])

    if debug:
        plt.figure(figsize=(8, 8))

        ax = plt.subplot(311)
        vlim = np.amax(np.abs(power))
        plt.pcolormesh(power, vmin=-vlim, vmax=vlim, cmap='coolwarm')
        plt.colorbar(label='power')
        plt.xlabel('Midpoint')
        plt.ylabel('Duration')

        plt.subplot(312, sharex=ax, sharey=ax)
        vlim = np.amax(np.abs(depth))
        plt.pcolormesh(depth, vmin=-vlim, vmax=vlim, cmap='coolwarm')
        plt.colorbar(label='depth scale')
        plt.xlabel('Midpoint')
        plt.ylabel('Duration')

        plt.subplot(313, sharex=ax, sharey=ax)
        plt.pcolormesh(npoints/min_points[:, np.newaxis], vmin=0, cmap='viridis')
        plt.colorbar(label='npoints/min_points')
        plt.xlabel('Midpoint')
        plt.ylabel('Duration')

        plt.tight_layout()
        plt.show()

    # Find the peak in the power, and associated dchisq values.
    irow = np.arange(power.shape[0])
    icol = np.argmax(power, axis=1)
    power_ = power[irow, icol]
    dchisq_dec_ = dchisq_dec[irow, icol]

    # Store the parameters corresponding to peak power.
    midpoints = period*(bin_edges[:-ncols] + bin_edges[ncols:])/2
    midpoint_vals_ = midpoints[icol]
    depth_vals_ = depth[irow, icol]
    flux_level_vals_ = flux_mean - depth_vals_ * gamma[irow, icol]

    power = np.full(nvals, np.nan)
    power[jmin:jmax] = power_

    dchisq_dec = np.full(nvals, np.nan)
    dchisq_dec[jmin:jmax] = dchisq_dec_

    dchisq_inc = np.full(nvals, np.nan)
    dchisq_inc[jmin:jmax] = dchisq_inc_

    midpoint_vals = np.full(nvals, np.nan)
    midpoint_vals[jmin:jmax] = midpoint_vals_

    depth_vals = np.full(nvals, np.nan)
    depth_vals[jmin:jmax] = depth_vals_

    flux_level_vals = np.full(nvals, np.nan)
    flux_level_vals[jmin:jmax] = flux_level_vals_

    return power, dchisq_dec, dchisq_inc, midpoint_vals, depth_vals, flux_level_vals


def _search_periods(periods, min_durations, max_durations, **kwargs):
    
    nrows = len(periods)
    ncols = len(kwargs['duration_grid'])
    power = np.full((nrows, ncols), np.nan)
    dchisq_dec = np.full((nrows, ncols), np.nan)
    dchisq_inc = np.full((nrows, ncols), np.nan)
    midpoint_vals = np.full((nrows, ncols), np.nan)
    depth_vals = np.full((nrows, ncols), np.nan)
    flux_level_vals = np.full((nrows, ncols), np.nan)

    search_func = partial(_search_period, **kwargs)
    for i, (period, min_duration, max_duration) in enumerate(zip(periods, min_durations, max_durations)):
        result = search_func(period, min_duration, max_duration)

        power[i] = result[0]
        dchisq_dec[i] = result[1]
        dchisq_inc[i] = result[2]
        midpoint_vals[i] = result[3]
        depth_vals[i] = result[4]
        flux_level_vals[i] = result[5]

    return power, dchisq_dec, dchisq_inc, midpoint_vals, depth_vals, flux_level_vals


def _search_periods_with_pool(num_processes, periods, min_durations, max_durations, **kwargs):

    nrows = len(periods)
    ncols = len(kwargs['duration_grid'])
    power = np.full((nrows, ncols), np.nan)
    dchisq_dec = np.full((nrows, ncols), np.nan)
    dchisq_inc = np.full((nrows, ncols), np.nan)
    midpoint_vals = np.full((nrows, ncols), np.nan)
    depth_vals = np.full((nrows, ncols), np.nan)
    flux_level_vals = np.full((nrows, ncols), np.nan)

    search_func = partial(_search_periods, **kwargs)
    with mp.Pool(processes=num_processes) as pool:
        
        i = 0
        period_chunks = [(periods[i::num_processes], min_durations[i::num_processes], max_durations[i::num_processes]) for i in range(num_processes)]
        for result in pool.starmap(search_func, period_chunks):
            power[i::num_processes, :] = result[0]
            dchisq_dec[i::num_processes, :] = result[1]
            dchisq_inc[i::num_processes, :] = result[2]
            midpoint_vals[i::num_processes, :] = result[3]
            depth_vals[i::num_processes, :] = result[4]
            flux_level_vals[i::num_processes, :] = result[5]

            i += 1

    return power, dchisq_dec, dchisq_inc, midpoint_vals, depth_vals, flux_level_vals


SearchResult = namedtuple('lstsq_result',
                          ['periods',
                           'period_groups',
                           'durations',
                           'duration_groups',
                           'power',
                           'chisq0',
                           'dchisq_dec',
                           'dchisq_inc',
                           'midpoint',
                           'duration',
                           'depth',
                           'flux_level',
                           'status_flag',
                           'best_period',
                           'best_midpoint',
                           'best_duration',
                           'best_depth',
                           'best_flux_level',
                           'model_phase',
                           'model_flux'])


def _prepare_lightcurve(flux: np.ndarray,
                        flux_err: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, float, float, float]:

    # Compute normalized weights.
    weights = 1 / flux_err ** 2
    weights_sum = np.sum(weights)
    weights_norm = weights/weights_sum

    # Compute some quantities.
    flux_mean = np.sum(weights_norm * flux)
    delta_flux_weighted = weights_norm * (flux - flux_mean)
    chisq0 = np.sum(weights_norm * (flux - flux_mean) ** 2)

    return weights_norm, delta_flux_weighted, weights_sum, flux_mean, chisq0


def _1d_periodogram(power,
                    dchisq_dec,
                    dchisq_inc,
                    midpoint_vals,
                    duration_grid,
                    depth_vals,
                    flux_level_vals,
                    duration_lims = None):

    # Apply specific duration limits.
    if duration_lims is not None:
        mask = ((duration_grid[np.newaxis, :] >= duration_lims.short[:, np.newaxis])
                & (duration_grid[np.newaxis, :] <= duration_lims.long[:, np.newaxis]))
        power = np.where(mask, power, np.nan)

    # Make a temporary array where NaNs are 0.
    nanmask = np.isnan(power)
    power0 = np.where(nanmask, 0, power)

    irow = np.arange(power0.shape[0])
    icol = np.argmax(power0, axis=1)

    power = power[irow, icol]
    dchisq_dec = dchisq_dec[irow, icol]
    dchisq_inc = dchisq_inc[irow, icol]
    midpoint_vals = midpoint_vals[irow, icol]
    duration_vals = duration_grid[icol]
    depth_vals = depth_vals[irow, icol]
    flux_level_vals = flux_level_vals[irow, icol]

    nanmask = np.all(nanmask, axis=1)
    midpoint_vals = np.where(nanmask, np.nan, midpoint_vals)
    duration_vals = np.where(nanmask, np.nan, duration_vals)
    depth_vals = np.where(nanmask, np.nan, depth_vals)
    flux_level_vals = np.where(nanmask, np.nan, flux_level_vals)

    return power, dchisq_dec, dchisq_inc, midpoint_vals, depth_vals, duration_vals, flux_level_vals


def template_lstsq(time: np.ndarray,
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
                   min_epoch_step: float = 60/SECINDAY,
                   max_epoch_step: float = 300/SECINDAY,
                   circular_orbits: bool = True,
                   frac_duration_step: float = 1.05,
                   normalisation: str = 'normal',
                   ld_type: str = 'linear',
                   ld_pars: tuple = (0.6,),
                   search_mode: str = 'TLS',
                   short_periods: str = 'skip',
                   smooth_window: Optional[float] = None,
                   smooth_weights: str = 'uniform',
                   max_duty_cycle: float = 0.2,
                   num_processes: Optional[int] = None,
                   diagnostic_plots: bool = False
                   ) -> tuple[SearchResult, SearchResult]:
    """ Perform a transit search with templates.
    """

    if search_mode not in ['BLS', 'TLS', 'WLS']:
        errmsg = f"Invalid value '{search_mode}' for parameter search_mode."
        raise ValueError(errmsg)

    if short_periods not in ['skip', 'TLS', 'WLS']:
        errmsg = f"Invalid value '{short_periods}' for parameter short_periods."
        raise ValueError(errmsg)

    if search_mode == 'WLS' and smooth_window is None:
        errmsg = f"Parameter smooth_window can not be None for WLS search."
        raise ValueError(errmsg)

    if search_mode != 'WLS' and smooth_window is not None:
        LOGWARNING(f"Performing {search_mode} search, setting smooth_window to None.")
        smooth_window = None

    if normalisation not in ['normal', 'dec_minus_inc']:
        errmsg = f"Invalid value '{normalisation}' for parameter normalisation."
        raise ValueError(errmsg)

    if smooth_weights not in ['uniform', 'tricube']:
        errmsg = f"Invalid value '{smooth_weights}' for parameter smooth_weights."
        raise ValueError(errmsg)

    # Pre-compute some arrays from the lightcurves.
    result = _prepare_lightcurve(flux, flux_err)
    weights_norm, delta_flux_weighted, weights_sum, flux_mean, chisq0 = result

    # Compute stellar densities from the stellar mass and radii ranges.
    min_stellar_density = SOLAR_DENSITY * min_stellar_mass / max_stellar_radius ** 3
    max_stellar_density = SOLAR_DENSITY * max_stellar_mass / min_stellar_radius ** 3
    stellar_density_bounds = (min_stellar_density, max_stellar_density)
    stellar_radius_bounds = (max_stellar_radius, min_stellar_radius)  # The max radius goes first because it matches the minimum density.

    # Compute the period grid to search.
    period_grid = grid.get_period_grid(max_stellar_density,
                                       np.ptp(time),
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
                                                           circular_orbits=False)

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
                                       exp_time,
                                       duration_lims,
                                       max_duty_cycle=max_duty_cycle,
                                       smooth_window=smooth_window)

    ngroups = len(period_groups)
    epoch_steps = np.zeros(ngroups)
    duration_groups = np.zeros_like(period_groups)
    for group in range(ngroups):

        imin, imax = period_groups[group]

        min_duration = np.amin(duration_lims.short[imin:imax])
        max_duration = np.amax(duration_lims.long[imin:imax])

        jmin = np.searchsorted(duration_grid, min_duration, side='left')
        jmax = np.searchsorted(duration_grid, max_duration, side='right')

        duration_groups[group, 0] = jmin
        duration_groups[group, 1] = jmax

        epoch_steps[group] = grid.get_epoch_step(min_duration,
                                                 epoch_sampling=epoch_sampling,
                                                 min_epoch_step=min_epoch_step,
                                                 max_epoch_step=max_epoch_step)

    # Set up variables for the output.
    nrows = period_grid.size
    ncols = duration_grid.size

    # TODO Making these and collapsing them after is more memory intensive?
    power = np.full((nrows, ncols), fill_value=np.nan)
    dchisq_dec = np.full((nrows, ncols), fill_value=np.nan)
    dchisq_inc = np.full((nrows, ncols), fill_value=np.nan)
    midpoint_vals = np.full((nrows, ncols), fill_value=np.nan)
    depth_vals = np.full((nrows, ncols), fill_value=np.nan)
    flux_level_vals = np.full((nrows, ncols), fill_value=np.nan)

    for group in range(ngroups):

        LOGDEBUG(f"Period group {group + 1} of {ngroups}:")

        epoch_step = epoch_steps[group]

        imin, imax = period_groups[group]
        period_group = period_grid[imin:imax]

        min_durations = duration_lims.short[imin:imax]
        max_durations = duration_lims.long[imin:imax]

        jmin, jmax = duration_groups[group]
        duration_group = duration_grid[jmin:jmax]

        search_mode_ = search_mode
        is_short_period = False
        baseline = np.amin(period_group) - np.amax(duration_group) - exp_time
        if search_mode == 'WLS' and baseline < smooth_window:
            if short_periods == 'skip':
                LOGINFO("  Skipping short periods in WLS search.")
                continue
            if short_periods == 'TLS':
                search_mode_ = 'TLS'
                LOGINFO("  Using TLS templates for short periods in WLS search.")
            if short_periods == 'WLS':
                search_mode_ = 'TLS'
                is_short_period = True
                LOGINFO("  Using WLS templates for short periods in WLS search.")

        LOGDEBUG(f"  Searching {len(period_group)} periods between {period_group[0]:.3f} days to {period_group[-1]:.3f} days.")
        LOGDEBUG(f"  Searching {len(duration_group)} durations between {duration_group[0]:.2f} days and {duration_group[-1]:.2f} days.")
        LOGDEBUG(f"  Searching using an epoch step of {epoch_step * SECINDAY / 60:.1f} minutes.")

        # Compute the template models for the current period set.
        templates = make_template_grid(period_group,
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
        kwargs['time'] = time
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
            result = _search_periods(period_group, min_durations, max_durations, **kwargs)
        else:
            result = _search_periods_with_pool(num_processes, period_group, min_durations, max_durations, **kwargs)

        power[imin:imax, jmin:jmax] = result[0]
        dchisq_dec[imin:imax, jmin:jmax] = result[1]
        dchisq_inc[imin:imax, jmin:jmax] = result[2]
        midpoint_vals[imin:imax, jmin:jmax] = result[3]
        depth_vals[imin:imax, jmin:jmax] = result[4]
        flux_level_vals[imin:imax, jmin:jmax] = result[5]

        # ipeak, jpeak = np.unravel_index(np.nanargmax(power), power.shape)
        # if ipeak >= imin:
        #     best_template_edges = templates[0]
        #     best_template_model = templates[1][jpeak - jmin]

    if diagnostic_plots:
        diagnostics.plot_2d_periodogram(period_grid, duration_grid, power, dchisq_dec, dchisq_inc, midpoint_vals, depth_vals, flux_level_vals, duration_circ, duration_full)

    result = _1d_periodogram(power, dchisq_dec, dchisq_inc, midpoint_vals, duration_grid, depth_vals, flux_level_vals, duration_circ)
    power_circ, dchisq_dec_circ, dchisq_inc_circ, midpoint_vals_circ, depth_vals_circ, duration_vals_circ, flux_level_vals_circ = result

    # Create status flags.
    status_flag = np.zeros_like(period_grid, dtype='uint8')

    # Save the peridogram.
    arg = np.nanargmax(power_circ)
    search_result_circ = SearchResult(periods=period_grid,
                                      period_groups=period_groups,
                                      durations=duration_grid,
                                      duration_groups=duration_groups,
                                      power=power_circ,
                                      chisq0=chisq0 * weights_sum,
                                      dchisq_dec=dchisq_dec_circ * weights_sum,
                                      dchisq_inc=dchisq_inc_circ * weights_sum,
                                      midpoint=midpoint_vals_circ,
                                      duration=duration_vals_circ,
                                      depth=depth_vals_circ,
                                      flux_level=flux_level_vals_circ,
                                      status_flag=status_flag,
                                      best_period=period_grid[arg],
                                      best_midpoint=midpoint_vals_circ[arg],
                                      best_duration=duration_vals_circ[arg],
                                      best_depth=depth_vals_circ[arg],
                                      best_flux_level=flux_level_vals_circ[arg],
                                      model_phase=None,
                                      model_flux=None)

    # templates = make_template_grid(best_period,
    #                                best_duration,
    #                                epoch_step,
    #                                exp_time,
    #                                exp_cadence,
    #                                ld_type=ld_type,
    #                                ld_pars=ld_pars,
    #                                search_mode=search_mode,
    #                                smooth_window=smooth_window,
    #                                smooth_weights=smooth_weights)
    #
    # evaluate_template(time, best_period, best_midpoint, edges, model)

    if diagnostic_plots:
        diagnostics.plot_1d_periodogram(search_result_circ)

    if circular_orbits:
        search_result_full = None
    else:
        result = _1d_periodogram(power, dchisq_dec, dchisq_inc, midpoint_vals, duration_grid, depth_vals, flux_level_vals, duration_full)
        power_full, dchisq_dec_full, dchisq_inc_full, midpoint_vals_full, depth_vals_full, duration_vals_full, flux_level_vals_full = result

        # Create status flags for each point in the full periodogram.
        status_flag = np.zeros_like(period_grid, dtype='uint8')
        status_flag = np.where(depth_vals_full > duration_circ.long, 1, status_flag)
        status_flag = np.where(depth_vals_full < duration_circ.short, 2, status_flag)

        # Save the peridogram.
        arg = np.nanargmax(power_full)
        search_result_full = SearchResult(periods=period_grid,
                                          period_groups=period_groups,
                                          durations=duration_grid,
                                          duration_groups=duration_groups,
                                          power=power_full,
                                          chisq0=chisq0 * weights_sum,
                                          dchisq_dec=dchisq_dec_full * weights_sum,
                                          dchisq_inc=dchisq_inc_full * weights_sum,
                                          midpoint=midpoint_vals_full,
                                          duration=duration_vals_full,
                                          depth=depth_vals_full,
                                          flux_level=flux_level_vals_full,
                                          status_flag=status_flag,
                                          best_period=period_grid[arg],
                                          best_midpoint=midpoint_vals_full[arg],
                                          best_duration=duration_vals_full[arg],
                                          best_depth=depth_vals_full[arg],
                                          best_flux_level=flux_level_vals_full[arg],
                                          model_phase=None,
                                          model_flux=None)

        if diagnostic_plots:
            diagnostics.plot_1d_periodogram(search_result_full)

    # Return the template model for the highest peak.
    # model_phase, model_flux = evaluate_template(time,
    #                                             best_period,
    #                                             best_midpoint,
    #                                             best_depth,
    #                                             best_flux_level,
    #                                             best_template_edges,
    #                                             best_template_model)

    return search_result_circ, search_result_full


def main():
    return


if __name__ == '__main__':
    main()
