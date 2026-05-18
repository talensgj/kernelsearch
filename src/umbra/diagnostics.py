import numpy as np

from . import grid, models

import matplotlib.pyplot as plt


def plot_duration_example():

    R_star = 1.0  # Solar radii.
    M_star = 1.0  # Solar masses.
    rho_sun = 1.41  # g cm^-3

    R_star_min = 0.8 * R_star
    R_star_max = 1.2 * R_star
    M_star_min = 0.8 * M_star
    M_star_max = 1.2 * M_star

    rho_min = rho_sun * M_star_min / R_star_max ** 3
    rho_max = rho_sun * M_star_max / R_star_min ** 3
    density_bounds = (rho_min, rho_max)
    stellar_radius_bounds = (R_star_max, R_star_min)  # R_star_max goes first since it matches rho_min.

    period_grid = grid.get_period_grid(rho_max, 90.)

    duration_ecc, _, _ = grid.get_transit_duration_limits(period_grid, density_bounds, stellar_radius_bounds)
    duration_circ, _, _ = grid.get_transit_duration_limits(period_grid, density_bounds, stellar_radius_bounds,
                                                      circular_orbits=True)

    min_duration = np.amin(duration_ecc.short)
    max_duration = np.amax(duration_ecc.long)

    duration_grid = grid.get_transit_duration_grid(min_duration, max_duration)

    plt.figure(figsize=(8, 5))

    plt.subplot(111, xscale='log', yscale='log')

    plt.fill_between(period_grid, duration_circ.short, duration_circ.long, facecolor='C0', edgecolor='k', alpha=0.5, label='circular')
    plt.fill_between(period_grid, duration_ecc.short, duration_circ.short, facecolor='C1', edgecolor='k', alpha=0.5)
    plt.fill_between(period_grid, duration_circ.long, duration_ecc.long, facecolor='C1', edgecolor='k', alpha=0.5, label='eccentric')

    plt.axhline(duration_grid[0], color='k', linestyle='--')
    plt.axhline(duration_grid[-1], color='k', linestyle='--')

    plt.xlim(0.3, 30.)

    plt.legend(loc='upper left')
    plt.xlabel('Period [days]')
    plt.ylabel('Duration [days]')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_power_at_period(phase_grid, duration_grid, power, depth, num_points, min_points):
    """ Make a diagnostic plot of the periodogram at a specific period.
    """

    # Depths in [ppm].
    depth = 1e6*depth

    plt.figure(figsize=(16, 8))

    ax = plt.subplot(311, yscale='log')

    vlim = np.amax(np.abs(power))
    plt.pcolormesh(phase_grid, duration_grid, power, vmin=-vlim, vmax=vlim, cmap='coolwarm')
    plt.colorbar(label='Power')

    plt.xlabel('Phase')
    plt.ylabel('Duration [days]')

    plt.subplot(312, sharex=ax, sharey=ax)

    vlim = np.amax(np.abs(depth))
    plt.pcolormesh(phase_grid, duration_grid, depth, vmin=-vlim, vmax=vlim, cmap='coolwarm')
    plt.colorbar(label='Depth [ppm]')

    plt.xlabel('Phase')
    plt.ylabel('Duration [days]')

    plt.subplot(313, sharex=ax, sharey=ax)

    plt.pcolormesh(phase_grid, duration_grid, num_points/min_points[:, np.newaxis], cmap='viridis')
    plt.colorbar(label='num_points/min_points')

    plt.xlabel('Phase')
    plt.ylabel('Duration [days]')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_lightcurve(time,
                    flux,
                    transit_model,
                    smooth_window=None):
    """ Make a diagnostic plot of the phase-folded lightcurve.
    """

    if smooth_window is None:
        smooth_window = 0

    parameters = transit_model['parameters']
    phase = models.phase_fold(time, parameters['period'], parameters['midpoint'])

    plt.figure(figsize=(12, 3))

    plt.subplot(111)

    plt.plot(phase, 1e6 * (flux - 1), '.', markersize=5, markeredgecolor='None', alpha=0.5, color='gray')
    plt.stairs(1e6 * (transit_model['flux'] - 1), transit_model['phase_edges'], baseline=None, lw=2, color='k', zorder=10)

    dx = 2*parameters['duration']/parameters['period'] + smooth_window/2
    plt.xlim(-dx, dx)

    plt.xlabel('Phase')
    plt.ylabel(r'$\Delta$Flux [ppm]')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_1d_periodogram(periodogram):

    plt.figure(figsize=(12, 15))

    ax = plt.subplot(711, xscale='log')

    plt.plot(periodogram['periods'], periodogram['power'])

    plt.ylabel('Power')

    plt.subplot(712, sharex=ax)

    plt.plot(periodogram['periods'], periodogram['dchisq_dec'], label=r'$\Delta\chi^2_{-}$')
    plt.plot(periodogram['periods'], periodogram['dchisq_inc'], label=r'$\Delta\chi^2_{+}$')

    plt.legend()
    plt.ylabel(r'$\Delta\chi^2$')

    plt.subplot(713, sharex=ax)
    plt.plot(periodogram['periods'], periodogram['num_points'])

    plt.ylabel('In-transit Points')

    plt.subplot(714, sharex=ax)
    plt.plot(periodogram['periods'], np.mod(periodogram['midpoint']/periodogram['periods'], 1))

    plt.ylabel('Phase')

    plt.subplot(715, yscale='log', sharex=ax)

    plt.plot(periodogram['periods'], periodogram['duration'])

    plt.plot(periodogram['periods'], periodogram['duration_short'], c='k')
    plt.plot(periodogram['periods'], periodogram['duration_long'], c='k')

    plt.ylabel('Duration [days]')

    plt.subplot(716, sharex=ax)

    plt.plot(periodogram['periods'], 1e6 * periodogram['depth'])

    plt.ylabel('Depth [ppm]')

    plt.subplot(717, sharex=ax)

    plt.plot(periodogram['periods'], 1e6 * (periodogram['flux_level'] - 1))

    plt.xlim(periodogram['periods'][0], periodogram['periods'][-1])

    plt.ylabel('Flux Level - 1 [ppm]')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_2d_periodogram(period_grid,
                        duration_grid,
                        power,
                        dchisq_dec,
                        dchisq_inc,
                        num_points,
                        midpoint_vals,
                        depth_vals,
                        flux_level_vals,
                        duration_circ,
                        duration_full):

    plt.figure(figsize=(12, 15))

    plt.subplot(711, xscale='log', yscale='log')

    plt.title('power')
    plt.pcolormesh(period_grid, duration_grid, power.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(712, xscale='log', yscale='log')
    plt.title('dchisq_dec')
    plt.pcolormesh(period_grid, duration_grid, dchisq_dec.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(713, xscale='log', yscale='log')
    plt.title('dchisq_inc')
    plt.pcolormesh(period_grid, duration_grid, dchisq_inc.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.subplot(714, xscale='log', yscale='log')
    plt.title('num_points')
    plt.pcolormesh(period_grid, duration_grid, num_points.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(715, xscale='log', yscale='log')
    plt.title('phase')
    plt.pcolormesh(period_grid, duration_grid, np.mod(midpoint_vals/period_grid[:, np.newaxis], 1).T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(716, xscale='log', yscale='log')
    plt.title('depth')
    plt.pcolormesh(period_grid, duration_grid, depth_vals.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(717, xscale='log', yscale='log')
    plt.title('flux_level')
    plt.pcolormesh(period_grid, duration_grid, flux_level_vals.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.xlabel('Period [days]')
    plt.ylabel('Duration [days]')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_oot_baseline(period_grid: np.ndarray,
                      baseline: np.ndarray,
                      smooth_window: float):
    """ Make a figure showing the OoT baseline.
    """

    plt.figure(figsize=(8, 5))

    plt.subplot(111)

    plt.plot(period_grid, baseline)

    plt.axhline(smooth_window, c='k')

    plt.xlim(period_grid[0], 2 * smooth_window)
    plt.ylim(0, 2 * smooth_window)

    plt.xlabel('Period [days]')
    plt.ylabel('OoT Baseline [days]')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_period_groups(period_grid: np.ndarray,
                       duration_lims: grid.DurationLimits,
                       intervals: list[tuple[int, int]],
                       icut: int):
    """ Make a figure showing the period groups.
    """

    plt.figure(figsize=(8, 5))

    ax = plt.subplot(211, xscale='log', yscale='log')

    plt.fill_between(period_grid, duration_lims.short, duration_lims.long, edgecolor='grey', alpha=0.5)

    for imin, imax in intervals:
        if imin == 0: continue

        if imin == icut:
            plt.axvline(period_grid[imin], c='r')
        else:
            plt.axvline(period_grid[imin], c='k')

    plt.xlabel('Period [days]')
    plt.ylabel('Duration [days]')

    plt.subplot(212, xscale='log', yscale='log', sharex=ax)

    plt.fill_between(period_grid, duration_lims.short / period_grid, duration_lims.long / period_grid, edgecolor='grey',
                     alpha=0.5)

    for imin, imax in intervals:
        if imin == 0: continue

        if imin == icut:
            plt.axvline(period_grid[imin], c='r')
        else:
            plt.axvline(period_grid[imin], c='k')

    plt.xlim(period_grid[0], period_grid[-1])

    plt.xlabel('Period [days]')
    plt.ylabel('Duty Cycle')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def main():
    return


if __name__ == '__main__':
    main()
