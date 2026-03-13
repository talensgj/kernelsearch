import numpy as np

HR_IN_DAY = 24
MIN_IN_DAY = 24*60
SEC_IN_DAY = 24*60*60


def bin_lightcurve(time: np.ndarray,
                   flux: np.ndarray,
                   flux_err: np.ndarray,
                   bin_size: float = 12,
                   method: str = 'points'
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Re-bin lightcurve.
    """

    if method == 'points':
        bin_idx = np.arange(len(time)) // bin_size
    elif method == 'window':
        nbins = np.ceil(np.ptp(time)/bin_size).astype('int')
        bin_edges = np.amin(time) + bin_size*np.arange(nbins + 1)
        bin_idx = np.searchsorted(bin_edges, time, side='right')
    else:
        raise ValueError(f"Unknown binning method: {method}.")

    weights = 1/flux_err**2

    points = np.bincount(bin_idx)
    time_sum = np.bincount(bin_idx, weights=time)
    weights_sum = np.bincount(bin_idx, weights=weights)
    weights_flux_sum = np.bincount(bin_idx, weights=weights*flux)

    mask = points > 0
    points = points[mask]
    time_sum = time_sum[mask]
    weights_sum = weights_sum[mask]
    weights_flux_sum = weights_flux_sum[mask]

    time = time_sum/points
    flux = weights_flux_sum/weights_sum
    flux_err = np.sqrt(1/weights_sum)

    # plt.plot(lc_data['time'][::10], lc_data['flux'][::10], '.')
    # plt.errorbar(time, flux, yerr=flux_err, ls='none')
    # plt.show()
    # plt.close()

    return time, flux, flux_err, points
