from typing import Optional
from collections import namedtuple
import multiprocessing as mp
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from transitleastsquares import grid, tls_constants

from . import models

SECINDAY = 24*3600


def evaluate_template(time,
                      period,
                      midpoint,
                      depth_scale,
                      flux_level,
                      template_edges,
                      template_model):

    phase = np.mod((time - midpoint) / period - 0.5, 1)  # Phase with transit at 0.5
    bin_idx = np.searchsorted(template_edges / period + 0.5, phase)  # Phase centered at 0.5
    template_model = np.append(np.append(0, template_model), 0)
    flux = depth_scale*template_model[bin_idx] + flux_level

    return phase, flux


def template_grid(period,
                  min_duration,
                  max_duration,
                  duration_step,
                  impact_params=(0., 0.6, 0.85, 0.95)):

    rp_rs = 0.05   # TODO hardcoded?

    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = period
    transit_params['R_p/R_s'] = rp_rs
    transit_params['a/R_s'] = 0.
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    # Initial time-step.
    time_step = 1/(12*24)  # TODO determine from ingress/egress duration.

    # Adjust the time-step so it is a multiple of the period.
    factor = period / time_step
    time_step = period / np.ceil(factor)

    # Adjust the duration values so the max duration is a multiple of the time-step.
    factor = max_duration / time_step
    factor = time_step * np.ceil(factor) / max_duration
    max_duration = factor * max_duration
    min_duration = factor * min_duration

    counter = (max_duration/time_step).astype('int')
    time_edges = np.linspace(-max_duration/2, max_duration/2, counter + 1)
    time = (time_edges[:-1] + time_edges[1:])/2

    flux_arr = []
    counter = np.ceil((max_duration - min_duration)/duration_step).astype('int')
    duration_vals = np.linspace(min_duration, max_duration, counter)
    for duration in duration_vals:
        for impact in impact_params:
            axis = models.duration2axis(duration, period, rp_rs, impact, 0., 90.)
            transit_params['a/R_s'] = axis
            transit_params['b'] = impact

            result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6], max_err=1)
            flux, nu, xp, yp, params, fac = result

            flux_arr.append(flux)

    flux_arr = np.row_stack(flux_arr)

    # time_edges = np.append(np.append(-np.inf, time_edges), np.inf)
    # flux_arr = np.column_stack([np.ones(len(flux_arr)), flux_arr, np.ones(len(flux_arr))])

    # plt.plot(time, flux_arr[-1,1:-1])
    # plt.show()
    #
    # plt.imshow(flux_arr)
    # plt.show()

    return time_edges, flux_arr - 1, time_step


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
        duration_grid.append(duration)

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
                       R_star_min,
                       R_star_max,
                       M_star_min,
                       M_star_max,
                       max_duty_cycle=0.15):

    imin = 0
    intervals = []
    for i, period in enumerate(periods):
        min_duration, max_duration = _get_duration_lims(period, R_star_min, R_star_max, M_star_min, M_star_max)
        if max_duration/periods[imin] > max_duty_cycle:
            intervals.append((imin, i))
            imin = i

    if imin != len(periods):
        intervals.append((imin, len(periods)))

    return intervals


def _make_kernel(mid_times,
                 duration_grid,
                 transit_params,
                 supersample_factor,
                 ld_type,
                 ld_pars,
                 exp_time):
    
    nrows = len(duration_grid)
    ncols = len(mid_times)
    template_count = np.zeros((nrows, ncols))
    template_models = np.zeros((nrows, ncols))
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
        template_count[row_idx] = result[0]

        # Evaluate the transit model.
        result = models.analytic_transit_model(mid_times,
                                               transit_params,
                                               ld_type,
                                               ld_pars,
                                               exp_time=exp_time,
                                               supersample_factor=supersample_factor,
                                               max_err=1.)
        
        template_models[row_idx] = result[0]
        
    return template_models, template_count
        
        
def _make_smooth_kernel(mid_times,
                        duration_grid,
                        transit_params,
                        supersample_factor,
                        ld_type,
                        ld_pars,
                        exp_time,
                        exp_cadence,
                        smooth_method,
                        smooth_window):

    nevals = np.ceil(smooth_window / exp_cadence).astype('int')
    if nevals % 2 == 0:
        nevals += 1

    mid_idx = nevals // 2
    dt = (np.arange(nevals) - mid_idx) * exp_cadence

    nrows = len(duration_grid)
    ncols = len(mid_times)
    template_count = np.zeros((nrows, ncols))
    template_models = np.zeros((nrows, ncols))
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
        template_count[row_idx] = result[0]

        result = models.analytic_transit_model(mid_times,
                                               transit_params,
                                               ld_type,
                                               ld_pars,
                                               exp_time=exp_time,
                                               supersample_factor=supersample_factor,
                                               max_err=1.)
        fac = result[5]

        for col_idx, mid_time in enumerate(mid_times):

            # Evaluate the transit model.
            result = models.analytic_transit_model(mid_time + dt,
                                                   transit_params,
                                                   ld_type,
                                                   ld_pars,
                                                   exp_time=exp_time,
                                                   supersample_factor=supersample_factor,
                                                   fac=fac,
                                                   max_err=1.)

            flux_dt = result[0]
            template_models[row_idx, col_idx] = flux_dt[mid_idx]/np.mean(flux_dt)
            
    return template_models, template_count


def make_template_grid(periods: np.ndarray,
                       duration_grid: np.ndarray,
                       bin_size: float,
                       exp_time: float,
                       exp_cadence: float,
                       ld_type: str = 'linear',
                       ld_pars: tuple = (0.6,),
                       ref_depth: float = 0.005,
                       smooth_method: str = 'mean',
                       smooth_window: Optional[float] = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    supersample_factor = np.ceil(exp_time*SECINDAY/10.).astype('int')

    if smooth_window is not None and smooth_window/np.amin(periods) > 0.5:  # TODO improve on this.
        print('Warning: smooth_window too long for shortest period, ignoring...')
        smooth_window = None

    ref_period = np.amax(periods)
    max_duration = np.amax(duration_grid)

    if smooth_window is None:
        delta_time = max_duration + exp_time
    else:
        delta_time = max_duration + np.maximum(exp_time, smooth_window)

    # Determine the times at which to evaluate the template.
    nbins = np.ceil(delta_time/bin_size).astype('int')
    template_edges = np.linspace(-delta_time/2, delta_time/2, nbins + 1)
    mid_times = (template_edges[:-1] + template_edges[1:])/2

    # Set up the transit parameters.
    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = ref_period
    transit_params['R_p/R_s'] = np.sqrt(ref_depth)
    transit_params['a/R_s'] = 0.
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    if smooth_window is None:
        template_models, template_count = _make_kernel(mid_times,
                                                       duration_grid,
                                                       transit_params,
                                                       supersample_factor,
                                                       ld_type,
                                                       ld_pars,
                                                       exp_time)
    else:
        template_models, template_count = _make_smooth_kernel(mid_times,
                                                              duration_grid,
                                                              transit_params,
                                                              supersample_factor,
                                                              ld_type,
                                                              ld_pars,
                                                              exp_time,
                                                              exp_cadence,
                                                              smooth_method,
                                                              smooth_window)

    template_count = (1 - template_count) > 0
    template_models = template_models - 1

    return template_edges, template_models, template_count


def _search_period(period,
                   time,
                   weights_norm,
                   delta_flux_weighted,
                   flux_mean,
                   chisq0,
                   bin_size,
                   template_models,
                   template_square,
                   template_count,
                   min_points,
                   debug=False
                   ):

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
        depth_scale = alpha / (beta - gamma ** 2)

    # Handle epoch/duration combinations with few or no in-transit points.
    # All elements of min_points must be >=1.
    min_points = np.maximum(min_points, 1)

    if np.isscalar(min_points):
        mask = npoints < min_points
    else:
        mask = npoints < min_points[:, np.newaxis]

    depth_scale[mask] = 0

    # Compute the delta chi-square.
    dchisq = alpha * depth_scale

    # Split delta chi-square by flux increaes and flux decreases.
    # Use flux increases to establish a baseline.
    select_inc = depth_scale < 0
    dchisq_inc = np.where(select_inc, dchisq, 0)
    dchisq_dec = np.where(select_inc, 0, dchisq)
    dchisq_inc = np.amax(dchisq_inc, axis=1)

    # Compute the power spectrum.
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
        vlim = np.amax(np.abs(depth_scale))
        plt.pcolormesh(depth_scale, vmin=-vlim, vmax=vlim, cmap='coolwarm')
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
    best_template_idx = imin + irow
    best_midpoint = period*(bin_edges[icol] + bin_edges[icol + ncols])/2
    best_depth_scale = depth_scale[irow, icol]
    best_flux_level = flux_mean - best_depth_scale * gamma[irow, icol]

    return power, dchisq_dec, dchisq_inc, best_template_idx, best_midpoint, best_depth_scale, best_flux_level


def _search_periods(periods, **kwargs):
    
    npoints = len(periods)
    power = np.zeros(npoints)
    dchisq_dec = np.zeros(npoints)
    dchisq_inc = np.zeros(npoints)
    best_template_idx = np.zeros(npoints, dtype='int')
    best_midpoint = np.zeros(npoints)
    best_depth_scale = np.zeros(npoints)
    best_flux_level = np.zeros(npoints)

    search_func = partial(_search_period, **kwargs)
    for i, period in enumerate(periods):
        result = search_func(period)
        power[i] = result[0]
        dchisq_dec[i] = result[1]
        dchisq_inc[i] = result[2]
        best_template_idx[i] = result[3]
        best_midpoint[i] = result[4]
        best_depth_scale[i] = result[5]
        best_flux_level[i] = result[6]
    
    return power, dchisq_dec, dchisq_inc, best_template_idx, best_midpoint, best_depth_scale, best_flux_level


def _search_periods_with_pool(num_processes, periods, **kwargs):
    
    npoints = len(periods)
    power = np.zeros(npoints)
    dchisq_dec = np.zeros(npoints)
    dchisq_inc = np.zeros(npoints)
    best_template_idx = np.zeros(npoints, dtype='int')
    best_midpoint = np.zeros(npoints)
    best_depth_scale = np.zeros(npoints)
    best_flux_level = np.zeros(npoints)
    
    search_func = partial(_search_periods, **kwargs)
    with mp.Pool(processes=num_processes) as pool:
        
        i = 0
        period_chunks = [periods[i::num_processes] for i in range(num_processes)]
        for result in pool.imap(search_func, period_chunks):
            power[i::num_processes] = result[0]
            dchisq_dec[i::num_processes] = result[1]
            dchisq_inc[i::num_processes] = result[2]
            best_template_idx[i::num_processes] = result[3]
            best_midpoint[i::num_processes] = result[4]
            best_depth_scale[i::num_processes] = result[5]
            best_flux_level[i::num_processes] = result[6]
            i += 1

    return power, dchisq_dec, dchisq_inc, best_template_idx, best_midpoint, best_depth_scale, best_flux_level


SearchResult = namedtuple('lstsq_result',
                          ['periods',
                           'power',
                           'chisq0',
                           'dchisq_dec',
                           'dchisq_inc',
                           'best_period',
                           'best_midpoint',
                           'best_duration',
                           'best_depth_scale',
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
                   smooth_method: str = 'mean',
                   smooth_window: Optional[float] = None,
                   min_bin_size: float = 1/(24*60),
                   max_bin_size: float = 5/(24*60),
                   oversampling_epoch: int = 3,
                   oversampling_duration: float = 4,
                   max_duty_cycle: float = 0.2,
                   num_processes: int = 1
                   ) -> SearchResult:
    """ Perform a transit search with templates.
    """

    # Pre-compute some arrays.
    result = _prepare_lightcurve(flux, flux_err)
    weights_norm, delta_flux_weighted, weights_sum, flux_mean, chisq0 = result

    # Group the periods for re-computing the kernels.
    periods = np.sort(periods)
    intervals = make_period_groups(periods,
                                   R_star_min,
                                   R_star_max,
                                   M_star_min,
                                   M_star_max,
                                   max_duty_cycle=max_duty_cycle)

    # Set up variables for the output.
    power = np.zeros_like(periods)
    dchisq_dec = np.zeros_like(periods)
    dchisq_inc = np.zeros_like(periods)
    best_period = np.nan
    best_midpoint = np.nan
    best_duration = np.nan
    best_depth_scale = np.nan
    best_flux_level = np.nan
    for imin, imax in intervals:

        # Get the duration grid for the current period set.
        bin_size, duration_grid = get_duration_grid(periods[imin:imax],
                                                    R_star_min,
                                                    R_star_max,
                                                    M_star_min,
                                                    M_star_max,
                                                    exp_time=exp_time,
                                                    min_bin_size=min_bin_size,
                                                    max_bin_size=max_bin_size,
                                                    oversampling_epoch=oversampling_epoch,
                                                    oversampling_duration=oversampling_duration)

        # Compute the template models for the current period set.
        template_edges, template_models, template_count = make_template_grid(periods[imin:imax],
                                                                             duration_grid,
                                                                             bin_size,
                                                                             exp_time,
                                                                             exp_cadence,
                                                                             ld_type=ld_type,
                                                                             ld_pars=ld_pars,
                                                                             smooth_method=smooth_method,
                                                                             smooth_window=smooth_window)
        template_square = template_models**2

        kwargs = dict()
        kwargs['time'] = time
        kwargs['weights_norm'] = weights_norm
        kwargs['delta_flux_weighted'] = delta_flux_weighted
        kwargs['flux_mean'] = flux_mean
        kwargs['chisq0'] = chisq0
        kwargs['bin_size'] = bin_size
        kwargs['template_models'] = template_models
        kwargs['template_square'] = template_square
        kwargs['template_count'] = template_count
        kwargs['min_points'] = 0.5*duration_grid/exp_cadence

        if num_processes is None:
            result = _search_periods(periods[imin:imax], **kwargs)
        else:
            result = _search_periods_with_pool(num_processes, periods[imin:imax], **kwargs)

        power[imin:imax] = result[0]
        dchisq_dec[imin:imax] = result[1]
        dchisq_inc[imin:imax] = result[2]

        ipeak = np.argmax(power)
        if ipeak >= imin:
            best_template_idx = result[3][ipeak - imin]
            best_period = periods[ipeak]
            best_midpoint = result[4][ipeak - imin]
            best_duration = duration_grid[best_template_idx]
            best_depth_scale = result[5][ipeak - imin]
            best_flux_level = result[6][ipeak - imin]
            best_template_edges = template_edges
            best_template_model = template_models[best_template_idx]

    # Return the template model for the highest peak.
    model_phase, model_flux = evaluate_template(time,
                                                best_period,
                                                best_midpoint,
                                                best_depth_scale,
                                                best_flux_level,
                                                best_template_edges,
                                                best_template_model)

    search_result = SearchResult(periods=periods,
                                 power=power,
                                 chisq0=chisq0*weights_sum,
                                 dchisq_dec=dchisq_dec*weights_sum,
                                 dchisq_inc=dchisq_inc*weights_sum,
                                 best_period=best_period,
                                 best_midpoint=best_midpoint,
                                 best_duration=best_duration,
                                 best_depth_scale=best_depth_scale,
                                 best_flux_level=best_flux_level,
                                 model_phase=model_phase,
                                 model_flux=model_flux)

    return search_result


def main():
    return


if __name__ == '__main__':
    main()
