import numpy as np

HR_IN_DAY = 24
MIN_IN_DAY = 24*60
SEC_IN_DAY = 24*60*60


def bin_lightcurve(time: np.ndarray,
                   flux: np.ndarray,
                   flux_err: np.ndarray,
                   bin_size: float = 12,
                   method: str = 'points'
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    val0 = np.bincount(bin_idx)
    val1 = np.bincount(bin_idx, weights=time)

    val2 = np.bincount(bin_idx, weights=weights)
    val3 = np.bincount(bin_idx, weights=weights*flux)

    mask = val0 > 0
    val0 = val0[mask]
    val1 = val1[mask]
    val2 = val2[mask]
    val3 = val3[mask]

    time = val1/val0
    flux = val3/val2
    flux_err = np.sqrt(1/val2)

    # plt.plot(lc_data['time'][::10], lc_data['flux'][::10], '.')
    # plt.errorbar(time, flux, yerr=flux_err, ls='none')
    # plt.show()
    # plt.close()

    return time, flux, flux_err
