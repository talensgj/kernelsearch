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


def _get_duration_lims(period):

    max_duration = grid.T14(
        R_s=tls_constants.R_STAR_MAX,
        M_s=tls_constants.M_STAR_MAX,
        P=period,
        small=False  # large planet for long transit duration
    )

    min_duration = grid.T14(
        R_s=tls_constants.R_STAR_MIN,
        M_s=tls_constants.M_STAR_MIN,
        P=period,
        small=True  # small planet for short transit duration
    )

    min_duration = min_duration*period
    max_duration = max_duration*period

    return min_duration, max_duration


def get_duration_lims(periods):

    min_duration0, max_duration0 = _get_duration_lims(np.amin(periods))
    min_duration1, max_duration1 = _get_duration_lims(np.amax(periods))

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
                      ref_depth: float = 0.005,
                      exp_time: Optional[float] = None,
                      min_bin_size: float = 1/(24*60),  # TODO are these good values?
                      max_bin_size: float = 5/(24*60),  # TODO are these good values?
                      oversampling_epoch: int = 3,
                      oversampling_duration: float = 4):

    ref_period = np.amax(periods)
    min_duration, max_duration = get_duration_lims(periods)

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


def make_period_groups(periods, max_duty_cycle=0.15):

    imin = 0
    intervals = []
    for i, period in enumerate(periods):
        min_duration, max_duration = _get_duration_lims(period)
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


def search_period(period,
                  time,
                  weights_norm,
                  delta_flux_weighted,
                  weights_sum,
                  flux_mean,
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

    # Compute a minimum dchisq by considering flux increases.
    select_inc = depth_scale < 0
    if np.any(select_inc):
        dchisq_inc = np.amax(dchisq[select_inc])
    else:
        dchisq_inc = 0

    if debug:
        plt.figure(figsize=(8, 8))

        ax = plt.subplot(311)
        tmp = weights_sum*(dchisq - dchisq_inc)
        vlim = np.amax(np.abs(tmp))
        plt.pcolormesh(tmp, vmin=-vlim, vmax=vlim, cmap='coolwarm')
        plt.colorbar(label=r'$\Delta \chi^2_{-} - \Delta \chi^2_{+}$')
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

    # Remove models that correspond to flux increases.
    dchisq[select_inc] = 0

    # Find the peak dchisq and store the best fit parameters.
    irow, icol = np.unravel_index(dchisq.argmax(), dchisq.shape)
    dchisq_dec = dchisq[irow, icol]
    best_template_idx = irow
    best_midpoint = period*(bin_edges[icol] + bin_edges[icol + ncols])/2
    best_depth_scale = depth_scale[irow, icol]
    best_flux_level = flux_mean - best_depth_scale * gamma[irow, icol]

    dchisq_dec = weights_sum * dchisq_dec
    dchisq_inc = weights_sum * dchisq_inc

    return dchisq_dec, dchisq_inc, best_template_idx, best_midpoint, best_depth_scale, best_flux_level


SearchResult = namedtuple('lstsq_result',
                          ['periods',
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
    chisq0 = weights_sum * np.sum(weights_norm * (flux - flux_mean) ** 2)

    return weights_norm, delta_flux_weighted, weights_sum, flux_mean, chisq0


def template_lstsq(time: np.ndarray,
                   flux: np.ndarray,
                   flux_err: np.ndarray,
                   periods: np.ndarray,
                   exp_time: float,
                   exp_cadence: float,
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
    intervals = make_period_groups(periods, max_duty_cycle=max_duty_cycle)

    # Set up variables for the output.
    dchisq_dec = np.zeros_like(periods)
    dchisq_inc = np.zeros_like(periods)
    best_period = np.nan
    best_midpoint = np.nan
    best_duration = np.nan
    best_depth_scale = np.nan
    best_flux_level = np.nan
    for imin, imax in intervals:

        # Create a pool for multiprocessing.
        with mp.Pool(processes=num_processes) as pool:

            # Get the duration grid for the current period set.
            bin_size, duration_grid = get_duration_grid(periods[imin:imax],
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

            # Set up muliprocessed period searches.
            params = partial(search_period,
                             time=time,
                             weights_norm=weights_norm,
                             delta_flux_weighted=delta_flux_weighted,
                             weights_sum=weights_sum,
                             flux_mean=flux_mean,
                             bin_size=bin_size,
                             template_models=template_models,
                             template_square=template_square,
                             template_count=template_count,
                             min_points=0.5*duration_grid/exp_cadence)

            # Do period searches.
            j = 0
            for result in pool.imap(params, periods[imin:imax]):

                # Update the results.
                dchisq_dec[imin+j] = result[0]
                dchisq_inc[imin+j] = result[1]

                if np.all(dchisq_dec[0:imin+j] < dchisq_dec[imin+j]):
                    best_template_idx = result[2]
                    best_period = periods[imin+j]
                    best_midpoint = result[3]
                    best_duration = duration_grid[best_template_idx]
                    best_depth_scale = result[4]
                    best_flux_level = result[5]
                    best_template_edges = template_edges
                    best_template_model = template_models[best_template_idx]

                j += 1

    # Return the template model for the highest peak.
    model_phase, model_flux = evaluate_template(time,
                                                best_period,
                                                best_midpoint,
                                                best_depth_scale,
                                                best_flux_level,
                                                best_template_edges,
                                                best_template_model)

    search_result = SearchResult(periods=periods,
                                 chisq0=chisq0,
                                 dchisq_dec=dchisq_dec,
                                 dchisq_inc=dchisq_inc,
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
