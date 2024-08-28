import numpy as np

from . import models


def evaluate_template(time,
                      period,
                      midpoint,
                      depth_scale,
                      template_time,
                      template_model):

    phase = np.mod((time - midpoint) / period - 0.5, 1)  # Phase with transit at 0.5
    bin_idx = np.searchsorted(template_time / period + 0.5, phase)  # Phase centered at 0.5
    template_model = np.append(np.append(0, template_model), 0)
    flux = depth_scale*template_model[bin_idx] + 1

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


def search_period(time,
                  flux,
                  flux_err,
                  period,
                  time_step,
                  template_time,
                  template_models,
                  epoch_search=False,
                  ):

    weights = 1/flux_err**2

    chi2 = np.sum(weights*(flux - 1)**2)
    chi2_ref = np.sum(weights*(flux - 1)**2)
    best_template = -1
    best_midpoint = np.nan
    best_depth_scale = np.nan

    phase = np.mod(time/period, 1)

    counter = (period/time_step).astype('int')
    bin_edges = np.linspace(0, 1, counter + 1)
    bin_idx = np.searchsorted(bin_edges, phase)
    a_bin = np.bincount(bin_idx, weights=weights, minlength=counter + 2)
    b_bin = np.bincount(bin_idx, weights=weights*(flux - 1), minlength=counter + 2)
    a_bin = a_bin[1:-1]
    b_bin = b_bin[1:-1]

    # Extend arrays.
    nrows, ncols = template_models.shape
    a_bin = np.append(a_bin, a_bin[:ncols-1])
    b_bin = np.append(b_bin, b_bin[:ncols-1])
    bin_edges = np.append(bin_edges, bin_edges[1:ncols] + 1)

    template_square = template_models**2

    for temp_idx in range(nrows):

        alpha_n = np.convolve(b_bin, template_models[temp_idx], 'valid')
        beta_n = np.convolve(a_bin, template_square[temp_idx], 'valid')

        depth_scale_n = alpha_n/beta_n
        chi2_n = chi2_ref - alpha_n**2/beta_n

        if epoch_search:
            midpoint = period * (bin_edges[:counter] + bin_edges[ncols:])/2
            return midpoint, chi2_n

        # TODO filter min depth?
        arg = np.argmin(chi2_n)
        if chi2_n[arg] < chi2:
            chi2 = chi2_n[arg]
            best_template = temp_idx
            best_midpoint = period*(bin_edges[arg] + bin_edges[arg + ncols + 1])/2
            best_depth_scale = depth_scale_n[arg]

    return chi2, best_template, best_midpoint, best_depth_scale


def template_lstsq(time,
                   flux,
                   flux_err,
                   periods,
                   min_duration,
                   max_duration,
                   template_time,
                   template_models):

    chi2 = np.zeros_like(periods)
    best_template = np.zeros_like(periods, dtype='int')
    best_midpoint = np.zeros_like(periods)
    best_depth_scale = np.zeros_like(periods)
    for i, period in enumerate(periods):
        print(i, period)

        result = template_grid(period,
                               min_duration,
                               max_duration,
                               2/(24*12))
        template_time, template_models, time_step = result

        result = search_period(time,
                               flux,
                               flux_err,
                               period,
                               time_step,
                               template_time,
                               template_models)

        chi2[i] = result[0]
        best_template[i] = result[1]
        best_midpoint[i] = result[2]
        best_depth_scale[i] = result[3]

    arg = np.argmin(chi2)
    phase, flux = evaluate_template(time, periods[arg], best_midpoint[arg], best_depth_scale[arg], template_time, template_models[best_template[arg]])

    return periods, chi2, best_template, best_midpoint, best_depth_scale, phase, flux


def main():
    return


if __name__ == '__main__':
    main()
