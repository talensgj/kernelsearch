from typing import Optional
from functools import partial
from collections import namedtuple

import numpy as np
from scipy import signal
import multiprocessing as mp

from kernelsearch import models
from transitleastsquares import grid, tls_constants

import matplotlib.pyplot as plt

SECINDAY = 24*3600


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


def _get_duration_lims(period,
                       R_star_min,
                       R_star_max,
                       M_star_min,
                       M_star_max):

    max_duration = grid.T14(
        R_s=R_star_max,
        M_s=M_star_max,
        P=period,
        small=False  # large planet for long transit duration
    )

    min_duration = grid.T14(
        R_s=R_star_min,
        M_s=M_star_min,
        P=period,
        small=True  # small planet for short transit duration
    )

    min_duration = min_duration*period
    max_duration = max_duration*period

    return min_duration, max_duration


def get_duration_lims(periods,
                      R_star_min,
                      R_star_max,
                      M_star_min,
                      M_star_max):

    min_duration0, max_duration0 = _get_duration_lims(np.amin(periods), R_star_min, R_star_max, M_star_min, M_star_max)
    min_duration1, max_duration1 = _get_duration_lims(np.amax(periods), R_star_min, R_star_max, M_star_min, M_star_max)

    min_duration = np.minimum(min_duration0, min_duration1)
    max_duration = np.maximum(max_duration0, max_duration1)

    return min_duration, max_duration


def _get_bin_size(min_duration,
                  ref_period,
                  ref_depth: float = 0.005,
                  exp_time: Optional[float] = None,
                  min_bin_size: float = 1 / (24 * 60),  # TODO are these good values?
                  max_bin_size: float = 5 / (24 * 60),  # TODO are these good values?
                  oversampling: int = 3):

    if exp_time is None:
        exp_time = 0.

    # Compute the approriate bin_size.
    axis = models.duration2axis(min_duration,
                                ref_period,
                                np.sqrt(ref_depth),
                                0., 0., 90.)
    full = models.axis2full(axis,
                            ref_period,
                            np.sqrt(ref_depth),
                            0., 0., 90.)

    ingress_time = (min_duration - full) / 2
    ingress_time = np.maximum(ingress_time, exp_time)
    bin_size = ingress_time / oversampling

    if bin_size < min_bin_size:
        bin_size = min_bin_size

    if bin_size > max_bin_size:
        bin_size = max_bin_size

    return bin_size


def _duration_grid(min_duration: float,
                   max_duration: float,
                   ref_period: float,
                   ref_depth: float = 0.005,
                   oversampling: float = 4):

    duration = min_duration
    duration_grid = [min_duration]
    while duration < max_duration:
        axis = models.duration2axis(duration,
                                    ref_period,
                                    np.sqrt(ref_depth),
                                    0., 0., 90.)
        full = models.axis2full(axis,
                                ref_period,
                                np.sqrt(ref_depth),
                                0., 0., 90.)

        # Increment by fractions of the previous transits ingress/egress.
        duration_step = (duration - full) / oversampling
        duration = duration + duration_step

        if duration < max_duration:
            duration_grid.append(duration)
        else:
            duration_grid.append(max_duration)

    duration_grid = np.array(duration_grid)

    return duration_grid


def get_duration_grid(periods: np.ndarray,
                      R_star_min: float,
                      R_star_max: float,
                      M_star_min: float,
                      M_star_max: float,
                      ref_depth: float = 0.005,
                      exp_time: Optional[float] = None,
                      min_bin_size: float = 1/(24*60),  # TODO are these good values?
                      max_bin_size: float = 5/(24*60),  # TODO are these good values?
                      oversampling_epoch: int = 3,
                      oversampling_duration: float = 4):

    ref_period = np.amax(periods)
    min_duration, max_duration = get_duration_lims(periods, R_star_min, R_star_max, M_star_min, M_star_max)

    bin_size = _get_bin_size(min_duration,
                             ref_period,
                             ref_depth=ref_depth,
                             exp_time=exp_time,
                             min_bin_size=min_bin_size,
                             max_bin_size=max_bin_size,
                             oversampling=oversampling_epoch)

    duration_grid = _duration_grid(min_duration,
                                   max_duration,
                                   ref_period,
                                   ref_depth=ref_depth,
                                   oversampling=oversampling_duration)

    return bin_size, duration_grid


def make_period_groups(periods,
                       exp_time,
                       R_star_min,
                       R_star_max,
                       M_star_min,
                       M_star_max,
                       smooth_window=None,
                       max_duty_cycle=0.15):

    imin = 0
    intervals = []
    if smooth_window is not None:
        # Guaranteed baseline if this period is the start of a period group.
        baseline = (1 - max_duty_cycle)*periods

        # Index of shortest period with baseline > smooth_window
        imin = np.searchsorted(baseline, smooth_window)

        if imin > 0:
            intervals = [(0, imin)]

    for i, period in enumerate(periods):
        min_duration, max_duration = _get_duration_lims(period, R_star_min, R_star_max, M_star_min, M_star_max)
        if (max_duration + exp_time)/periods[imin] > max_duty_cycle:
            intervals.append((imin, i))
            imin = i

    if imin != len(periods):
        intervals.append((imin, len(periods)))

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
                       bin_size: float,
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
        print("Warning: Cannot make WLS templates for this period range, defaulting to TLS templates.")
        search_mode = 'TLS'

    if search_mode in ['BLS', 'TLS']:
        delta_time = max_duration + exp_time
    else:
        delta_time = max_duration + exp_time + smooth_window

    if periods.size == 1:
        delta_time = np.minimum(delta_time, max_period)

    # Determine the times at which to evaluate the template.
    nbins = np.ceil(delta_time/bin_size).astype('int')
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
                   time,
                   weights_norm,
                   delta_flux_weighted,
                   flux_mean,
                   chisq0,
                   bin_size,
                   star_kwargs,
                   duration_grid,
                   templates,
                   min_points,
                   normalisation,
                   debug=False
                   ):

    # Select the durations that encompass the required range at this period.
    min_duration, max_duration = _get_duration_lims(period, **star_kwargs)
    imin = np.searchsorted(duration_grid, min_duration)
    imax = np.searchsorted(duration_grid, max_duration)
    imin = np.maximum(imin - 1, 0)
    imax = np.minimum(imax, len(duration_grid))

    min_points = min_points[imin:imax]
    duration_grid = duration_grid[imin:imax]

    template_edges = templates[0]
    template_models = templates[1][imin:imax]
    template_square = templates[2][imin:imax]
    template_count = templates[3][imin:imax]

    # nrows: number of kernels (i.e. durations), ncols: length of transit kernels.
    nrows, ncols = template_models.shape

    # Phase fold the data.
    phase = np.mod(time/period, 1)

    # Create the phase bins.
    num_bins = np.ceil(period/bin_size).astype('int')
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
    dchisq_inc = np.amax(dchisq_inc, axis=1)

    # Compute the power spectrum.
    if normalisation == 'normal':
        power = dchisq_dec/chisq0
    else:
        power = (dchisq_dec - dchisq_inc[:, np.newaxis])/(chisq0 - dchisq_inc[:, np.newaxis])

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
    irow, icol = np.unravel_index(power.argmax(), power.shape)
    power = power[irow, icol]
    dchisq_dec = dchisq_dec[irow, icol]
    dchisq_inc = dchisq_inc[irow]

    # Store the parameters corresponding to peak power.
    best_midpoint = period*(bin_edges[icol] + bin_edges[icol + ncols])/2
    best_duration = duration_grid[irow]
    best_depth = depth[irow, icol]
    best_flux_level = flux_mean - best_depth * gamma[irow, icol]
    best_edges = template_edges
    best_model = template_models[irow]

    return power, dchisq_dec, dchisq_inc, best_midpoint, best_duration, best_depth, best_flux_level, best_edges, best_model


def _search_periods(periods, **kwargs):
    
    npoints = len(periods)
    power = np.zeros(npoints)
    dchisq_dec = np.zeros(npoints)
    dchisq_inc = np.zeros(npoints)
    best_midpoint = np.zeros(npoints)
    best_duration = np.zeros(npoints)
    best_depth = np.zeros(npoints)
    best_flux_level = np.zeros(npoints)
    best_edges = None
    best_model = None

    peak_power = -np.inf
    search_func = partial(_search_period, **kwargs)
    for i, period in enumerate(periods):
        result = search_func(period)

        power[i] = result[0]
        dchisq_dec[i] = result[1]
        dchisq_inc[i] = result[2]
        best_midpoint[i] = result[3]
        best_duration[i] = result[4]
        best_depth[i] = result[5]
        best_flux_level[i] = result[6]

        if power[i] > peak_power:
            peak_power = power[i]
            best_edges = result[7]
            best_model = result[8]
    
    return power, dchisq_dec, dchisq_inc, best_midpoint, best_duration, best_depth, best_flux_level, best_edges, best_model


def _search_periods_with_pool(num_processes, periods, **kwargs):
    
    npoints = len(periods)
    power = np.zeros(npoints)
    dchisq_dec = np.zeros(npoints)
    dchisq_inc = np.zeros(npoints)
    best_midpoint = np.zeros(npoints)
    best_duration = np.zeros(npoints)
    best_depth = np.zeros(npoints)
    best_flux_level = np.zeros(npoints)
    best_edges = None
    best_model = None

    peak_power = -np.inf
    search_func = partial(_search_periods, **kwargs)
    with mp.Pool(processes=num_processes) as pool:
        
        i = 0
        period_chunks = [periods[i::num_processes] for i in range(num_processes)]
        for result in pool.imap(search_func, period_chunks):
            power[i::num_processes] = result[0]
            dchisq_dec[i::num_processes] = result[1]
            dchisq_inc[i::num_processes] = result[2]
            best_midpoint[i::num_processes] = result[3]
            best_duration[i::num_processes] = result[4]
            best_depth[i::num_processes] = result[5]
            best_flux_level[i::num_processes] = result[6]

            if np.amax(result[0]) > peak_power:
                peak_power = np.amax(result[0])
                best_edges = result[7]
                best_model = result[8]

            i += 1

    return power, dchisq_dec, dchisq_inc, best_midpoint, best_duration, best_depth, best_flux_level, best_edges, best_model


SearchResult = namedtuple('lstsq_result',
                          ['periods',
                           'period_groups',
                           'durations',
                           'duration_groups',
                           'power',
                           'chisq0',
                           'dchisq_dec',
                           'dchisq_inc',
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


def template_lstsq(time: np.ndarray,
                   flux: np.ndarray,
                   flux_err: np.ndarray,
                   periods: np.ndarray,
                   exp_time: float,
                   exp_cadence: float,
                   R_star_min: float = tls_constants.R_STAR_MIN,
                   R_star_max: float = tls_constants.R_STAR_MAX,
                   M_star_min: float = tls_constants.M_STAR_MIN,
                   M_star_max: float = tls_constants.M_STAR_MAX,
                   ld_type: str = 'linear',
                   ld_pars: tuple = (0.6,),
                   search_mode: str = 'TLS',
                   normalisation: str = 'normal',
                   smooth_window: Optional[float] = None,
                   smooth_weights: str = 'uniform',
                   min_bin_size: float = 1 / (24 * 60),
                   max_bin_size: float = 5 / (24 * 60),
                   oversampling_epoch: int = 3,
                   oversampling_duration: float = 4,
                   max_duty_cycle: float = 0.2,
                   num_processes: Optional[int] = None
                   ) -> SearchResult:
    """ Perform a transit search with templates.
    """

    if search_mode not in ['BLS', 'TLS', 'WLS']:
        errmsg = f"Invalid value '{search_mode}' for parameter search_mode."
        raise ValueError(errmsg)

    if search_mode == 'WLS' and smooth_window is None:
        errmsg = f"Parameter smooth_window can not be None for WLS search."
        raise ValueError(errmsg)

    if normalisation not in ['normal', 'dec_minus_inc']:
        errmsg = f"Invalid value '{normalisation}' for parameter normalisation."
        raise ValueError(errmsg)

    if smooth_weights not in ['uniform', 'tricube']:
        errmsg = f"Invalid value '{smooth_weights}' for parameter smooth_weights."
        raise ValueError(errmsg)

    # Make sure period grid is sorted.
    periods = np.sort(periods)

    # Pre-compute some arrays.
    result = _prepare_lightcurve(flux, flux_err)
    weights_norm, delta_flux_weighted, weights_sum, flux_mean, chisq0 = result

    # Group the periods for re-computing the kernels.
    period_groups = make_period_groups(periods,
                                       exp_time,
                                       R_star_min,
                                       R_star_max,
                                       M_star_min,
                                       M_star_max,
                                       smooth_window=smooth_window,
                                       max_duty_cycle=max_duty_cycle)

    ngroups = len(period_groups)
    bin_sizes = np.zeros(ngroups)
    durations = np.array([])
    duration_groups = np.zeros_like(period_groups)
    for group in range(ngroups):

        imin, imax = period_groups[group]
        period_grid = periods[imin:imax]

        # Get the duration grid for the current period group.
        bin_size, duration_grid = get_duration_grid(period_grid,
                                                    R_star_min,
                                                    R_star_max,
                                                    M_star_min,
                                                    M_star_max,
                                                    exp_time=exp_time,
                                                    min_bin_size=min_bin_size,
                                                    max_bin_size=max_bin_size,
                                                    oversampling_epoch=oversampling_epoch,
                                                    oversampling_duration=oversampling_duration)

        bin_sizes[group] = bin_size
        duration_groups[group, 0] = durations.size
        durations = np.concatenate([durations, duration_grid])
        duration_groups[group, 1] = durations.size

    # Set up variables for the output.
    power = np.zeros_like(periods)
    dchisq_dec = np.zeros_like(periods)
    dchisq_inc = np.zeros_like(periods)
    best_period = np.nan
    best_midpoint = np.nan
    best_duration = np.nan
    best_depth = np.nan
    best_flux_level = np.nan

    for group in range(ngroups):

        bin_size = bin_sizes[group]

        imin, imax = period_groups[group]
        period_grid = periods[imin:imax]

        jmin, jmax = duration_groups[group]
        duration_grid = durations[jmin:jmax]

        # Compute the template models for the current period set.
        templates = make_template_grid(period_grid,
                                       duration_grid,
                                       bin_size,
                                       exp_time,
                                       exp_cadence,
                                       ld_type=ld_type,
                                       ld_pars=ld_pars,
                                       search_mode=search_mode,
                                       smooth_window=smooth_window,
                                       smooth_weights=smooth_weights)

        kwargs = dict()
        kwargs['time'] = time
        kwargs['weights_norm'] = weights_norm
        kwargs['delta_flux_weighted'] = delta_flux_weighted
        kwargs['flux_mean'] = flux_mean
        kwargs['chisq0'] = chisq0
        kwargs['bin_size'] = bin_size
        kwargs['star_kwargs'] = {'R_star_min': R_star_min,
                                 'R_star_max': R_star_max,
                                 'M_star_min': M_star_min,
                                 'M_star_max': M_star_max}
        kwargs['duration_grid'] = duration_grid
        kwargs['templates'] = templates
        kwargs['min_points'] = 0.5*duration_grid/exp_cadence
        kwargs['normalisation'] = normalisation

        if num_processes is None:
            result = _search_periods(period_grid, **kwargs)
        else:
            result = _search_periods_with_pool(num_processes, period_grid, **kwargs)

        power[imin:imax] = result[0]
        dchisq_dec[imin:imax] = result[1]
        dchisq_inc[imin:imax] = result[2]

        ipeak = np.argmax(power)
        if ipeak >= imin:
            best_period = periods[ipeak]
            best_midpoint = result[3][ipeak - imin]
            best_duration = result[4][ipeak - imin]
            best_depth = result[5][ipeak - imin]
            best_flux_level = result[6][ipeak - imin]
            best_template_edges = result[7]
            best_template_model = result[8]

    # Return the template model for the highest peak.
    model_phase, model_flux = evaluate_template(time,
                                                best_period,
                                                best_midpoint,
                                                best_depth,
                                                best_flux_level,
                                                best_template_edges,
                                                best_template_model)

    search_result = SearchResult(periods=periods,
                                 period_groups=period_groups,
                                 durations=durations,
                                 duration_groups=duration_groups,
                                 power=power,
                                 chisq0=chisq0*weights_sum,
                                 dchisq_dec=dchisq_dec*weights_sum,
                                 dchisq_inc=dchisq_inc*weights_sum,
                                 best_period=best_period,
                                 best_midpoint=best_midpoint,
                                 best_duration=best_duration,
                                 best_depth=best_depth,
                                 best_flux_level=best_flux_level,
                                 model_phase=model_phase,
                                 model_flux=model_flux)

    return search_result


def main():
    return


if __name__ == '__main__':
    main()
