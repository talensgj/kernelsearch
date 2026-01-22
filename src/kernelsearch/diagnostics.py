import numpy as np

import matplotlib.pyplot as plt


def plot_1d_periodogram(periodogram):

    plt.figure(figsize=(12, 15))

    plt.subplot(611, xscale='log')

    plt.plot(periodogram.periods, periodogram.power)

    plt.ylabel('Power')

    plt.subplot(612, xscale='log')

    plt.plot(periodogram.periods, periodogram.dchisq_dec)
    plt.plot(periodogram.periods, periodogram.dchisq_inc)

    plt.ylabel('dchisq')

    plt.subplot(613, xscale='log')
    plt.plot(periodogram.periods, periodogram.midpoint/periodogram.periods)

    plt.ylabel('Phase')

    plt.subplot(614, xscale='log')

    plt.plot(periodogram.periods, periodogram.duration)

    plt.ylabel('Duration')

    plt.subplot(615, xscale='log')

    plt.plot(periodogram.periods, periodogram.depth)

    plt.ylabel('Depth')

    plt.subplot(616, xscale='log')

    plt.plot(periodogram.periods, periodogram.flux_level)

    plt.ylabel('Flux Level')

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
    plt.pcolormesh(period_grid, duration_grid, (midpoint_vals / period_grid[:, np.newaxis]).T)

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


def main():
    return


if __name__ == '__main__':
    main()
