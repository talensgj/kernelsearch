from typing import Optional

import numpy as np

from transitleastsquares import grid, tls_constants

from . import models

import matplotlib.pyplot as plt

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


def get_duration_grid(periods: np.ndarray,
                      ref_depth: float = 0.005,
                      exp_time: Optional[float] = None,
                      min_bin_size: float = 1/(24*60),  # TODO are these good values?
                      max_bin_size: float = 5/(24*60),  # TODO are these good values?
                      oversampling: int = 3):

    if exp_time is None:
        exp_time = 0.

    ref_period = np.amax(periods)
    min_duration, max_duration = get_duration_lims(periods)

    # Compute the approriate bin_size.
    axis = models.duration2axis(min_duration,
                                ref_period,
                                np.sqrt(ref_depth),
                                0., 0., 90.)
    full = models.axis2full(axis,
                            ref_period,
                            np.sqrt(ref_depth),
                            0., 0., 90.)

    ingress_time = (min_duration - full)/2
    ingress_time = np.maximum(ingress_time, exp_time)
    bin_size = ingress_time/oversampling

    if bin_size < min_bin_size:
        bin_size = min_bin_size

    if bin_size > max_bin_size:
        bin_size = max_bin_size

    # duration_step = oversampling*bin_size
    # nvals = np.ceil((max_duration - min_duration)/duration_step).astype('int')  # TODO Use bin_size or (fraction of) ingress_time?
    # duration_grid = min_duration + duration_step*np.arange(nvals)

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
        duration_step = (duration - full)/4  # Increment by 1/2 of the previous transits egress.
        duration = duration + duration_step
        duration_grid.append(duration)

    duration_grid = np.array(duration_grid)

    return bin_size, duration_grid


def make_template_grid(periods: np.ndarray,
                       duration_grid: np.ndarray,
                       bin_size: float,
                       ld_type: str = 'linear',
                       ld_pars: tuple = (0.6,),
                       ref_depth: float = 0.005,
                       exp_time: Optional[float] = None):

    if exp_time is None:
        exp_time = 0.
        supersample_factor = 1
    else:
        supersample_factor = np.ceil(exp_time*SECINDAY/10.).astype('int')

    ref_period = np.amax(periods)
    max_duration = np.amax(duration_grid)
    delta_time = max_duration + exp_time

    # Determine the times at which to evaluate the template.
    nbins = np.ceil(delta_time/bin_size).astype('int')
    template_edges = np.linspace(-delta_time/2, delta_time/2, nbins + 1)
    time = (template_edges[:-1] + template_edges[1:])/2

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

    nrows = len(duration_grid)
    ncols = len(time)
    template_models = np.zeros((nrows, ncols))
    for row_idx, duration in enumerate(duration_grid):

        # Compute the scaled semi-major axis that gives the required duration.
        axis = models.duration2axis(duration,
                                    ref_period,
                                    np.sqrt(ref_depth),
                                    0.,
                                    0.,
                                    90.)
        transit_params['a/R_s'] = axis

        # Evaluate the transit model.
        result = models.analytic_transit_model(time,
                                               transit_params,
                                               ld_type,
                                               ld_pars,
                                               exp_time=exp_time,
                                               supersample_factor=supersample_factor,
                                               max_err=1.)
        template_models[row_idx] = result[0] - 1

    return template_edges, template_models


def search_period(time,
                  flux,
                  weights,
                  period,
                  bin_size,
                  template_models,
                  epoch_search=False,
                  ):

    flux_mean = np.sum(weights*flux)/np.sum(weights)

    a = np.sum(weights)
    phase = np.mod(time/period, 1)
    counter = (period/bin_size).astype('int')
    bin_edges = np.linspace(0, 1, counter + 1)
    bin_idx = np.searchsorted(bin_edges, phase)
    a_bin = np.bincount(bin_idx, weights=weights, minlength=counter + 2)
    b_bin = np.bincount(bin_idx, weights=weights*(flux - flux_mean), minlength=counter + 2)
    a_bin = a_bin[1:-1]
    b_bin = b_bin[1:-1]

    # Extend arrays.
    nrows, ncols = template_models.shape
    a_bin = np.append(a_bin, a_bin[:ncols-1])
    b_bin = np.append(b_bin, b_bin[:ncols-1])
    bin_edges = np.append(bin_edges, bin_edges[1:ncols] + 1)

    template_square = template_models**2

    best_dchisq = 0
    best_template = -1
    best_midpoint = np.nan
    best_depth_scale = np.nan
    best_flux_level = np.nan
    for temp_idx in range(nrows):

        alpha = np.convolve(b_bin, template_models[temp_idx], 'valid')
        beta = np.convolve(a_bin, template_square[temp_idx], 'valid')
        gamma = np.convolve(a_bin, template_models[temp_idx], 'valid')

        # Compute the depth scaling and keep only flux decreases.
        depth_scale = a*alpha/(a*beta - gamma**2)
        depth_scale = np.where(depth_scale < 0, 0, depth_scale)

        # Compute the delta chi-square.
        dchisq = alpha*depth_scale

        if epoch_search:
            midpoint = period * (bin_edges[:counter] + bin_edges[ncols:])/2
            flux_level = flux_mean - depth_scale*gamma/a
            return midpoint, dchisq, depth_scale, flux_level

        arg = np.argmax(dchisq)
        if dchisq[arg] > best_dchisq:
            best_dchisq = dchisq[arg]
            best_template = temp_idx
            best_midpoint = period*(bin_edges[arg] + bin_edges[arg + ncols])/2
            best_depth_scale = depth_scale[arg]
            best_flux_level = flux_mean - depth_scale[arg]*gamma[arg]/a

    return best_dchisq, best_template, best_midpoint, best_depth_scale, best_flux_level


def template_lstsq(time,
                   flux,
                   flux_err,
                   periods,
                   exp_time: Optional[float] = None,
                   min_bin_size: float = 1/(24*60),
                   max_bin_size: float = 5/(24*60),
                   oversampling: int = 3):

    weights = 1/flux_err**2

    # Get the duration grid.
    bin_size, duration_grid = get_duration_grid(periods,
                                                exp_time=exp_time,
                                                min_bin_size=min_bin_size,
                                                max_bin_size=max_bin_size,
                                                oversampling=oversampling)

    # Compute the template models.
    template_edges, template_models = make_template_grid(periods,
                                                         duration_grid,
                                                         bin_size=bin_size,
                                                         exp_time=exp_time)

    dchisq = np.zeros_like(periods)
    best_template = np.zeros_like(periods, dtype='int')
    best_midpoint = np.zeros_like(periods)
    best_depth_scale = np.zeros_like(periods)
    best_flux_level = np.zeros_like(periods)
    for i, period in enumerate(periods):
        print(i)

        # Get the duration limits at this period.
        min_duration, max_duration = _get_duration_lims(period)

        # Select the templates to use at this period.
        imin = np.searchsorted(duration_grid, min_duration)
        imax = np.searchsorted(duration_grid, max_duration)
        imin = np.maximum(imin - 1, 0)
        imax = np.minimum(imax + 1, len(duration_grid) - 1)

        # Perform the transit search.
        result = search_period(time,
                               flux,
                               weights,
                               period,
                               bin_size,
                               template_models[imin:imax])

        # Update the results.
        dchisq[i] = result[0]
        best_template[i] = result[1] + imin
        best_midpoint[i] = result[2]
        best_depth_scale[i] = result[3]
        best_flux_level[i] = result[4]

    # Return the template model for the highest peak.
    arg = np.argmax(dchisq)
    phase, model = evaluate_template(time,
                                     periods[arg],
                                     best_midpoint[arg],
                                     best_depth_scale[arg],
                                     best_flux_level[arg],
                                     template_edges,
                                     template_models[best_template[arg]])

    return periods, dchisq, best_template, best_midpoint, best_depth_scale, phase, model


def main():
    return


if __name__ == '__main__':
    main()
