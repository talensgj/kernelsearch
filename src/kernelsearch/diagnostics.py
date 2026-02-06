import numpy as np

from . import grid

import matplotlib.pyplot as plt


def plot_1d_periodogram(periodogram,
                        duration_circ,
                        duration_full):

    plt.figure(figsize=(12, 15))

    ax = plt.subplot(611, xscale='log')

    plt.plot(periodogram.periods, periodogram.power)

    plt.ylabel('Power')

    plt.subplot(612, sharex=ax)

    plt.plot(periodogram.periods, periodogram.dchisq_dec, label=r'$\Delta\chi^2_{-}$')
    plt.plot(periodogram.periods, periodogram.dchisq_inc, label=r'$\Delta\chi^2_{+}$')

    plt.legend()
    plt.ylabel(r'$\Delta\chi^2$')

    plt.subplot(613, sharex=ax)
    plt.plot(periodogram.periods, np.mod(periodogram.midpoint/periodogram.periods, 1))

    plt.ylabel('Phase')

    plt.subplot(614, yscale='log', sharex=ax)

    plt.plot(periodogram.periods, periodogram.duration)

    plt.plot(periodogram.periods, duration_full.short, c='k')
    plt.plot(periodogram.periods, duration_full.long, c='k')

    plt.plot(periodogram.periods, duration_circ.short, c='k', ls='--')
    plt.plot(periodogram.periods, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(615, sharex=ax)

    plt.plot(periodogram.periods, 1e6 * periodogram.depth)

    plt.ylabel('Depth [ppm]')

    plt.subplot(616, sharex=ax)

    plt.plot(periodogram.periods, 1e6 * (periodogram.flux_level - 1))

    plt.xlim(periodogram.periods[0], periodogram.periods[-1])

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
                        midpoint_vals,
                        depth_vals,
                        flux_level_vals,
                        duration_circ,
                        duration_full):

    plt.figure(figsize=(12, 15))

    plt.subplot(611, xscale='log', yscale='log')

    plt.title('power')
    plt.pcolormesh(period_grid, duration_grid, power.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(612, xscale='log', yscale='log')
    plt.title('dchisq_dec')
    plt.pcolormesh(period_grid, duration_grid, dchisq_dec.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(613, xscale='log', yscale='log')
    plt.title('dchisq_inc')
    plt.pcolormesh(period_grid, duration_grid, dchisq_inc.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(614, xscale='log', yscale='log')
    plt.title('phase')
    plt.pcolormesh(period_grid, duration_grid, np.mod(midpoint_vals/period_grid[:, np.newaxis], 1).T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(615, xscale='log', yscale='log')
    plt.title('depth')
    plt.pcolormesh(period_grid, duration_grid, depth_vals.T)

    plt.plot(period_grid, duration_full.short, c='k')
    plt.plot(period_grid, duration_full.long, c='k')

    plt.plot(period_grid, duration_circ.short, c='k', ls='--')
    plt.plot(period_grid, duration_circ.long, c='k', ls='--')

    plt.ylabel('Duration [days]')

    plt.subplot(616, xscale='log', yscale='log')
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
