import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd

import wotan
import transitleastsquares as transitlstsq

from src import models, search

import matplotlib.pyplot as plt

# Global variables.
RNG = np.random.default_rng(5627323756)
SECINDAY = 24*3600


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


def make_test_lightcurve(length: float,
                         period: float,
                         depth: float,
                         duration: float,
                         noise_ppm: float = 1000.,
                         ld_type: str = 'linear',
                         ld_pars: tuple = (0.6,),
                         exp_time: float = 0.,
                         supersampling: int = 1):

    # Compute the scaled semi-major axis.
    axis = models.duration2axis(duration,
                                period,
                                np.sqrt(depth),
                                0.,
                                0.,
                                90.)

    # A simple transit model.
    transit_params = dict()
    transit_params['T_0'] = period*RNG.random()
    transit_params['P'] = period
    transit_params['R_p/R_s'] = np.sqrt(depth)
    transit_params['a/R_s'] = axis
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    npoints = np.ceil(length / exp_time).astype('int')
    time = np.arange(npoints) * exp_time
    result = models.analytic_transit_model(time, transit_params, ld_type, ld_pars, max_err=1, exp_time=exp_time,
                                           supersample_factor=supersampling)

    flux = result[0] + RNG.normal(size=npoints)*noise_ppm/1e6
    flux_err = np.ones_like(flux)*noise_ppm/1e6

    return time, flux, flux_err


def make_single_template(period: float,
                         duration: float,
                         bin_size: float,
                         ld_type: str = 'linear',
                         ld_pars: tuple = (0.6,),
                         ref_depth: float = 0.005,
                         exp_time: float = 0.,
                         supersampling: int = 1):

    # Compute the scaled semi-major axis.
    axis = models.duration2axis(duration,
                                period,
                                np.sqrt(ref_depth),
                                0.,
                                0.,
                                90.)

    # Make the transit model
    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = period
    transit_params['R_p/R_s'] = np.sqrt(ref_depth)
    transit_params['a/R_s'] = axis
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    npoints = np.ceil((duration + exp_time) / bin_size).astype('int')
    template_edges = np.linspace(-(duration + exp_time) / 2, (duration + exp_time) / 2, npoints + 1)
    time = (template_edges[:-1] + template_edges[1:]) / 2

    result = models.analytic_transit_model(time, transit_params, ld_type, ld_pars, max_err=1, exp_time=exp_time,
                                           supersample_factor=supersampling)
    template_model = result[0] - 1

    return template_edges, template_model


def test_steps():

    true_period = 3.4849
    true_duration = 3.4/24

    time, flux, flux_err = make_test_lightcurve(30.,
                                                true_period,
                                                0.005,
                                                true_duration,
                                                exp_time=300./SECINDAY,
                                                supersampling=30)
    flux = flux*2
    flux_err = flux_err*2
    weights = 1/flux_err**2

    plt.figure(figsize=(13, 5))

    plt.plot(time, flux, '.')

    plt.xlabel('Time [days]')
    plt.ylabel('Flux')

    plt.tight_layout()
    plt.show()

    # variations = [(30./SECINDAY, 0., 1),
    #               (30./SECINDAY, 300./SECINDAY, 30),
    #               (30./SECINDAY, 1800./SECINDAY, 180),
    #               (30./SECINDAY, 3600./SECINDAY, 360)]

    variations = [(30./SECINDAY, 300./SECINDAY, 30),
                  (60./SECINDAY, 300./SECINDAY, 30),
                  (120./SECINDAY, 300./SECINDAY, 30),
                  (300./SECINDAY, 300./SECINDAY, 30),
                  (600./SECINDAY, 300./SECINDAY, 30)]

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for bin_size, exp_time, supersampling in variations:

        # Scale bin_size to be a multiple of the period.
        factor = true_period/bin_size
        bin_size = true_period/np.ceil(factor)

        # Make the template fit in an integer number of bins.
        time_edges, template = make_single_template(true_period, true_duration, bin_size, exp_time=exp_time, supersampling=supersampling)
        template = np.array([template])

        # Perform the epoch search.
        start = timer()
        x, y, z, flux_n = search.search_period(time, flux, weights, true_period, time_step=bin_size, template_models=template, epoch_search=True)
        runtime = timer() - start
        print(bin_size, runtime)

        # Fix the epoch range.
        x = np.where(x > true_period, x - true_period, x)
        sort = np.argsort(x)
        x = x[sort]
        y = y[sort]
        z = z[sort]
        flux_n = flux_n[sort]

        ax1.plot(x, y)

        phase, model = search.evaluate_template(time, true_period, x[np.argmax(y)], z[np.argmax(y)], flux_n[np.argmax(y)], time_edges, template[0])
        ax2.plot(phase, flux, 'k.')
        ax2.plot(phase, model, '.')

    ax1.set_xlabel('Epoch [days]')
    ax1.set_ylabel('$\Delta \chi^2$')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_steps()
