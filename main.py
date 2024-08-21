import numpy as np
from astropy import units

from src import models
import src.search

import matplotlib.pyplot as plt


def main():

    m_star_min = 0.59*units.solMass  # K9V
    m_star_max = 1.61*units.solMass  # F0V

    r_star_min = 0.61*units.solRad  # K9V
    r_star_max = 1.73*units.solRad  # F0V

    rho_star_min = 0.5  # TODO
    rho_star_max = 3.0  # TODO

    period_min = 130.
    period_max = 150.

    grid_size = 1001

    period_x = np.linspace(period_min, period_max, grid_size)
    a_r_star_lower = models.density2axis(rho_star_min, period_x)
    a_r_star_upper = models.density2axis(rho_star_max, period_x)

    duration_upper = models.axis2duration(a_r_star_lower, period_x, 0.1, 0., 0., 90.)
    duration_lower = models.axis2duration(a_r_star_upper, period_x, 0.1, 0., 0., 90.)

    period = np.random.rand(grid_size)*(period_max - period_min) + period_min
    density = np.random.rand(grid_size)*(rho_star_max - rho_star_min) + rho_star_min
    a_r_star = models.density2axis(density, period)
    r_planet = np.random.rand(grid_size)*(0.2 - 0.005) + 0.005
    ecc = np.random.rand(grid_size)*(0.5 - 0.0) + 0.0
    w = np.random.rand(grid_size)*360
    b = np.random.rand(grid_size)*(0.5 - 0) + 0
    duration = models.axis2duration(a_r_star, period, 0.1, b, ecc, w)

    plt.subplot(211)
    plt.plot(period_x, a_r_star_lower)
    plt.plot(period_x, a_r_star_upper)
    plt.plot(period, a_r_star, 'k.')

    plt.xlabel('Period [days]')
    plt.ylabel('a [R_star]')

    plt.subplot(212)
    plt.plot(period_x, duration_lower, c='C0')
    plt.plot(period_x, 0.5 * duration_lower, c='C0', ls='--')
    plt.plot(period_x, duration_upper, c='C1')
    plt.plot(period_x, 2.0 * duration_upper, c='C1', ls='--')
    plt.plot(period, duration, 'k.')

    plt.xlabel('Period [days]')
    plt.ylabel('T14/P [hours]')

    plt.show()

    duration_max = np.amax(duration)
    time = np.linspace(-duration_max/2, duration_max/2, 101)
    flux_arr = np.zeros((grid_size, len(time)))
    for i in range(grid_size):
        transit_params = dict()
        transit_params['T_0'] = 0.
        transit_params['P'] = period[i]
        transit_params['R_p/R_s'] = 0.1
        transit_params['a/R_s'] = a_r_star[i]
        transit_params['b'] = b[i]
        transit_params['ecc'] = ecc[i]
        transit_params['w'] = w[i]
        transit_params['Omega'] = 0.

        result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6],)
        flux, nu, xp, yp, params, fac = result
        flux_arr[i] = flux

    duration = np.linspace(np.amin(duration), np.amax(duration), 101)
    a_r_star = models.duration2axis(duration, 25., 0.1, 0., 0., 90.)

    duration_max = np.amax(duration)
    time1 = np.linspace(-duration_max / 2, duration_max / 2, 101)
    flux_arr1 = np.zeros((101, len(time)))
    for i in range(101):
        transit_params = dict()
        transit_params['T_0'] = 0.
        transit_params['P'] = 25.
        transit_params['R_p/R_s'] = 0.1
        transit_params['a/R_s'] = a_r_star[i]
        transit_params['b'] = 0.
        transit_params['ecc'] = 0.
        transit_params['w'] = 90.
        transit_params['Omega'] = 0.

        result = models.analytic_transit_model(time1, transit_params, ld_type='linear', ld_pars=[0.6], )
        flux, nu, xp, yp, params, fac = result
        flux_arr1[i] = flux

    plt.plot(time, flux_arr.T, c='k')
    plt.plot(time1, flux_arr1.T, c='r')
    plt.show()

    return


def depth_scaling():

    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = 25.
    transit_params['R_p/R_s'] = 0.1
    transit_params['a/R_s'] = 30.
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    time = np.linspace(-0.5, 0.5, 101)

    result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6])
    flux1, nu, xp, yp, params, fac = result

    transit_params['R_p/R_s'] = 0.15
    result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6])
    flux2, nu, xp, yp, params, fac = result

    transit_params['R_p/R_s'] = 0.05
    result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6])
    flux3, nu, xp, yp, params, fac = result

    plt.plot(time, flux1-1)
    plt.plot(time, (flux2-1)/(0.15/0.10)**2)
    plt.plot(time, (flux3-1)/(0.05/0.10)**2)
    plt.show()


def transit_grid(period, min_duration, max_duration, nsamples=10, impact_params=[0., 0.6, 0.85, 0.95]):

    val = models.axis2duration(10., 5., 0.1, 0.9, 0.4, 230)
    val = models.duration2axis(val, 5., 0.1, 0.9, 0.4, 230)
    print(val)

    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = period
    transit_params['R_p/R_s'] = 0.1
    transit_params['a/R_s'] = 0.
    transit_params['b'] = 0.
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    time = np.linspace(-max_duration/2, max_duration/2)

    for duration in np.linspace(min_duration, max_duration, nsamples):
        for impact in impact_params:
            axis = models.duration2axis(duration, period, 0.1, impact, 0., 90.)
            transit_params['a/R_s'] = axis
            transit_params['b'] = impact

            result = models.analytic_transit_model(time, transit_params, ld_type='linear', ld_pars=[0.6])
            flux, nu, xp, yp, params, fac = result

            plt.plot(time, flux)

    plt.show()


if __name__ == '__main__':
    # main()
    # depth_scaling()
    # transit_grid(25, 5., 12.)
    # src.search.test()
    src.search.real_test()

