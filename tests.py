import os

import numpy as np
import pandas as pd

import wotan
import transitleastsquares as transitlstsq

from src import models, search

import matplotlib.pyplot as plt


def test():

    true_period = 3.556
    true_midpoint = 1.2
    true_duration = 0.24

    template_time, template_models, time_step = search.template_grid(true_period, 0.2, 0.5, 1/(12*24))

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

    result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6], max_err=1)
    flux, nu, xp, yp, params, fac = result

    # template_time = np.array([-np.inf, -0.06, -0.03, 0.03, 0.06, np.inf])
    # template_models = np.array([[0, -0.5, -0.5, -0.5, 0],
    #                             [0, 0.5, 0, -0.5, 0],
    #                             [0, -0.5, 0, 0.5, 0]])

    result = search.search_period(time,
                                  flux,
                                  flux_err,
                                  period=true_period,
                                  time_step=time_step,
                                  template_time=template_time,
                                  template_models=template_models)
    chi2, best_template, best_midpoint, best_depth_scale = result
    print(result)

    phase, model = search.evaluate_template(time, true_period, best_midpoint, best_depth_scale, template_time, template_models[best_template])

    plt.plot(time, flux)
    plt.plot(time, model)
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
                                                  window_length=2.1,
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

    min_duration = 0.10996734401247639
    max_duration = 1.4531496279883123
    template_time, template_models, _ = search.template_grid(140.,
                                                             min_duration,
                                                             max_duration,
                                                             2 / (24 * 12),
                                                             impact_params=(0, 0.95))

    print(template_models.shape)
    plt.imshow(template_models)
    plt.show()

    flux_biweight_err = flux_biweight_err/np.nanmean(flux_biweight_err)  # Same as TLS
    result = search.template_lstsq(time[mask],
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
    sort = np.argsort(phase)
    phase = phase[sort]
    flux = flux_biweight[mask][sort]
    model = model[sort]

    plt.figure(figsize=(8, 5))

    plt.subplot(211)
    plt.plot(tls_result.periods, tls_result.chi2)
    plt.plot(my_periods, my_chi2)

    plt.xlabel('Period [days]')
    plt.ylabel('Chi2')

    plt.subplot(212)
    plt.plot(phase, flux, 'k.')
    plt.plot(phase, model, lw=3)

    plt.xlim(-max_duration/140.+0.5, max_duration/140.+0.5)

    plt.xlabel('Phase')
    plt.ylabel('Flux')

    plt.tight_layout()
    plt.savefig('template_test.png', dpi=300)
    plt.show()


def test_steps():

    exp_time = 3600./(24*3600)
    supersample_factor = 360

    # A simple transit model.
    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = 3.3847
    transit_params['R_p/R_s'] = 0.1
    transit_params['a/R_s'] = 10.
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    duration = models.axis2duration(transit_params['a/R_s'],
                                    transit_params['P'],
                                    transit_params['R_p/R_s'],
                                    transit_params['b'],
                                    transit_params['ecc'],
                                    transit_params['w'])

    npoints = np.ceil(10/exp_time).astype('int')
    time = np.arange(npoints)*exp_time
    result = models.analytic_transit_model(time, transit_params, 'linear', [0.6], max_err=1, exp_time=exp_time, supersample_factor=supersample_factor)
    flux = result[0] + np.random.randn(npoints)*1e-3
    flux_err = 1e-3*np.ones_like(flux)

    plt.plot(time, flux, '.')
    plt.show()

    variations = [(30./(24*3600), 0., 1),
                  (30./(24*3600), 300./(24*3600), 30),
                  (30./(24*3600), 1800./(24*3600), 180),
                  (30./(24*3600), 3600./(24*3600), 360)]
                  # (300. / (24 * 3600), 0., 1),
                  # (300. / (24 * 3600), 300. / (24 * 3600), 30),
                  # (300. / (24 * 3600), 1800. / (24 * 3600), 180),
                  # (1800. / (24 * 3600), 0., 1),
                  # (1800. / (24 * 3600), 300. / (24 * 3600), 30),
                  # (1800. / (24 * 3600), 1800. / (24 * 3600), 180)
                  # ]
    for bin_size, exp_time, supersample_factor in variations:
        bin_size = (duration+exp_time)/np.ceil((duration + exp_time)/bin_size)
        print(bin_size, (duration+exp_time)/bin_size)
        npoints = ((duration+exp_time)/bin_size).astype('int')
        time_edges = np.linspace(-(duration + exp_time)/2, (duration + exp_time)/2, npoints + 1)
        time_ = (time_edges[:-1] + time_edges[1:])/2

        result = models.analytic_transit_model(time_, transit_params, 'linear', [0.6], max_err=1, exp_time=exp_time, supersample_factor=supersample_factor)

        # plt.plot(time_*24, result[0], '.')
        # plt.show()

        template_models = np.array([result[0] - 1])
        print(template_models.shape)

        x, y = search.search_period(time, flux, flux_err, transit_params['P'], time_step=bin_size, template_time=time_edges, template_models=template_models, epoch_search=True)

        plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    test_steps()
