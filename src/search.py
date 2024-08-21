import os

import pandas as pd
import wotan
import transitleastsquares as transitlstsq
import numpy as np
import matplotlib.pyplot as plt

from . import models


def evaluate_template(time,
                      period,
                      midpoint,
                      depth_scale,
                      template_time,
                      template_model):

    phase = np.mod((time - midpoint) / period - 0.5, 1)  # Phase with transit at 0.5

    bin_idx = np.searchsorted(template_time / period + 0.5, phase)  # Phase centered at 0.5
    flux = depth_scale*template_model[bin_idx] + 1

    return phase, flux


def search_period(time,
                  flux,
                  flux_err,
                  period,
                  epoch_step,
                  max_duration,
                  template_time,
                  template_models
                  ):

    weights = 1/flux_err**2

    chi2 = np.sum(weights*(flux - 1)**2)
    chi2_ref = np.sum(weights*(flux - 1)**2)
    best_template = -1
    best_midpoint = np.nan
    best_depth_scale = np.nan

    phase = np.mod(time/period - 0.5, 1)

    # Extend arrays by 1 duration.
    mask = phase < max_duration/period
    phase = np.append(phase, phase[mask] + 1)
    flux = np.append(flux, flux[mask])
    weights = np.append(weights, weights[mask])

    sort = np.argsort(phase)
    phase = phase[sort]
    flux = flux[sort]
    weights = weights[sort]

    tmp_arr = weights*(flux - 1)

    template_phase = template_time / period + 0.5
    template_phase[0] = 0
    template_phase[-1] = 1
    template_square = template_models**2

    epoch_step = template_time[2] - template_time[1]
    counter = np.ceil(period/epoch_step).astype('int')
    midpoint_vals = np.linspace(-period/2 + max_duration/2, (period + max_duration)/2, counter)
    for midpoint in midpoint_vals:

        loop_phase = np.copy(template_phase)
        loop_phase[1:-1] = loop_phase[1:-1] + midpoint/period
        if midpoint >= 0:  # TODO
            loop_phase[0] = max_duration/period
            loop_phase[-1] = 1 + max_duration/period

        # plt.plot(phase, flux - 1, 'k.')
        # plt.axvline(loop_phase[0], c='C0')
        # plt.axvline(loop_phase[-1], c='C0')
        # plt.plot(loop_phase[1:-1], template_models[:,1:-1].T)
        # plt.show()
        # plt.close()

        bin_idx = np.searchsorted(loop_phase, phase)  # Phase centered at 0.5

        _, ncols = template_models.shape
        Abin = np.bincount(bin_idx, weights=weights, minlength=ncols + 2)
        Bbin = np.bincount(bin_idx, weights=tmp_arr, minlength=ncols + 2)
        Abin = Abin[1:-1]
        Bbin = Bbin[1:-1]

        alpha_n = np.sum(template_models*Bbin, axis=1)
        beta_n = np.sum(template_square*Abin, axis=1)

        depth_scale_n = alpha_n/beta_n
        chi2_n = chi2_ref - alpha_n**2/beta_n
        # TODO filter min depth?
        arg = np.argmin(chi2_n)
        if chi2_n[arg] < chi2:
            chi2 = chi2_n[arg]
            best_template = arg
            best_midpoint = midpoint
            best_depth_scale = depth_scale_n[arg]

    return chi2, best_template, best_midpoint, best_depth_scale


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

    counter = np.ceil(max_duration*(12*24)).astype('int')
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

            result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6])
            flux, nu, xp, yp, params, fac = result

            flux_arr.append(flux)

    flux_arr = np.row_stack(flux_arr)

    time = np.append(np.append(-np.inf, time), np.inf)
    flux_arr = np.column_stack([np.ones(len(flux_arr)), flux_arr, np.ones(len(flux_arr))])

    # plt.imshow(flux_arr)
    # plt.show()

    return time_edges, flux_arr - 1


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
        nfolds = np.ceil(np.ptp(time)/period).astype('int')

        result = search_period(time,
                               flux,
                               flux_err,
                               period,
                               min_duration / 10 / nfolds,
                               max_duration,
                               template_time,
                               template_models)

        chi2[i] = result[0]
        best_template[i] = result[1]
        best_midpoint[i] = result[2]
        best_depth_scale[i] = result[3]

    arg = np.argmin(chi2)
    phase, flux = evaluate_template(time, periods[arg], best_midpoint[arg], best_depth_scale[arg], template_time, template_models[arg])

    return periods, chi2, best_template, best_midpoint, best_depth_scale, phase, flux


def test():

    true_period = 3.5
    true_midpoint = 1.2
    true_duration = 0.24

    template_time, template_models = template_grid(true_period, 0.2, 0.5, 1/(12*24))

    time = np.arange(5000)*1/(24*12)
    phase = np.mod((time - true_midpoint)/true_period, 1)
    flux = np.where(np.abs(phase) < true_duration/true_period/2, 0.99, 1)
    flux_err = 0.001*np.ones_like(flux)

    transit_params = dict()
    transit_params['T_0'] = true_midpoint
    transit_params['P'] = true_period
    transit_params['R_p/R_s'] = 0.07
    transit_params['a/R_s'] = models.duration2axis(true_duration, true_period, 0.07, 0.2, 0.05, 160.)
    transit_params['b'] = 0.2
    transit_params['ecc'] = 0.05
    transit_params['w'] = 160.
    transit_params['Omega'] = 0.

    result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6])
    flux, nu, xp, yp, params, fac = result

    # template_time = np.array([-np.inf, -0.06, -0.03, 0.03, 0.06, np.inf])
    # template_models = np.array([[0, -0.5, -0.5, -0.5, 0],
    #                             [0, 0.5, 0, -0.5, 0],
    #                             [0, -0.5, 0, 0.5, 0]])

    result = search_period(time,
                           flux,
                           flux_err,
                           period=true_period,
                           epoch_step=true_duration/20,
                           max_duration=0.5,
                           template_time=template_time,
                           template_models=template_models)
    chi2, best_template, best_midpoint, best_depth_scale = result
    print(result)
    phase = np.mod((time - best_midpoint) / true_period - 0.5, 1)  # Phase with transit at 0.5
    bin_idx = np.searchsorted(template_time / true_period + 0.5, phase)  # Phase centered at 0.5
    bin_idx = bin_idx - 1

    model = template_models[best_template, bin_idx]
    plt.plot(time, flux - 1)
    plt.plot(time, best_depth_scale*model)
    plt.show()


def bin_lightcurve(lc_data, bin_size=12):
    """ Re-bin a plato lightcurve based on a cadence.
    """

    bin_idx = np.arange(len(lc_data)) // bin_size

    weights = 1/lc_data['flux_err']**2

    val0 = np.bincount(bin_idx)
    val1 = np.bincount(bin_idx, weights=lc_data['time'])
    val2 = np.bincount(bin_idx, weights=weights*lc_data['flux'])
    val3 = np.bincount(bin_idx, weights=weights)

    time = val1/val0
    flux = val2/val3
    flux_err = np.sqrt(1/val3)

    # plt.plot(lc_data['time'][::10], lc_data['flux'][::10], '.')
    # plt.errorbar(time, flux, yerr=flux_err, ls='none')
    # plt.show()
    # plt.close()

    return time, flux, flux_err


def real_test():

    lc_id = 99
    lc_dir = '/home/talens/Research/PLATO/plato_simulations/WP111000_simulations/simulations_Breton/complete_lightcurve_habitable'

    # Read the lightcurve.
    lc_file = os.path.join(lc_dir, f'{lc_id * 100:010d}.ftr')
    lc_data = pd.read_feather(lc_file)

    # Bin to 5-min resolution.
    time, flux, flux_err = bin_lightcurve(lc_data)

    # Use a simple biweight detrending.
    flux_biweight, trend_biweight = wotan.flatten(time,
                                                  flux,
                                                  method='biweight',
                                                  window_length=0.75,
                                                  edge_cutoff=0.5,
                                                  break_tolerance=0.5,
                                                  cval=5,
                                                  return_trend=True)
    flux_biweight_err = flux_err / trend_biweight

    # Remove bad points.
    mask = np.isfinite(trend_biweight)
    tls = transitlstsq.transitleastsquares(time[mask], flux_biweight[mask], flux_biweight_err[mask], verbose=True)
    tls_result = tls.power(period_min=135,
                           period_max=145,
                           n_transits_min=1,
                           use_threads=8)

    min_duration = 0.15
    max_duration = 0.90
    template_time, template_models = template_grid(140.,
                                                   min_duration,
                                                   max_duration,
                                                   2/(24*12),
                                                   impact_params=(0, 0.95))

    print(template_models.shape)
    plt.imshow(template_models)
    plt.show()

    flux_biweight_err = flux_biweight_err/np.nanmean(flux_biweight_err)  # Same as TLS
    result = template_lstsq(time[mask],
                            flux_biweight[mask],
                            flux_biweight_err[mask],
                            tls_result.periods,
                            min_duration,
                            max_duration,
                            template_time,
                            template_models)

    my_periods = result[0]
    my_chi2 = result[1]
    phase = result[5]
    model = result[6]

    plt.figure(figsize=(8, 5))

    plt.subplot(211)
    plt.plot(tls_result.periods, tls_result.chi2)
    plt.plot(my_periods, my_chi2)

    plt.xlabel('Period [days]')
    plt.ylabel('Chi2')

    plt.subplot(212)
    plt.plot(phase, flux_biweight[mask], 'k.')
    plt.plot(phase, model)

    plt.xlim(-max_duration/140.+0.5, max_duration/140.+0.5)

    plt.xlabel('Phase')
    plt.ylabel('Flux')

    plt.tight_layout()
    plt.savefig('template_test.png', dpi=300)
    plt.show()
