from typing import Optional

import numpy as np
from scipy import signal, interpolate

import wotan
import wotan.gaps

import matplotlib.pyplot as plt


def get_wotan_kwargs(method: str) -> dict:

    wotan_kwargs = dict()
    wotan_kwargs['edge_cutoff'] = 0.  # Don't discard data.
    wotan_kwargs['break_tolerance'] = 0.5  # Treat quarters individually.

    if method == 'biweight':
        wotan_kwargs['cval'] = 5

    return wotan_kwargs


def filter_wotan(time: np.ndarray,
                 flux: np.ndarray,
                 flux_err: np.ndarray,
                 method: str,
                 window_length: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Filter a lightcurve using wotan.

    Parameters
    ----------
    time: np.ndarray
       Array of observation times.
    flux: np.ndarray
        Array of flux measurements corresponding to the observation times.
    flux_err: np.ndarray
        Array of flux errors corresponding to the observation times.
    method: str
        The filtering method to use.
    window_length: float
        The size of the smoothing window used with wotan, must have the same units as time.

    Returns
    -------
    time: np.ndarray
        Array of observation times, may be shorter than the input.
    flux_detrend: np.ndarray
        Detrended flux values corresponding to time.
    flux_detrend_error: np.ndarray
        Detrended error values corresponding to time.
    trend: np.ndarray
        The trend that was removed from the flux and error values.

    """

    wotan_kwargs = get_wotan_kwargs(method)

    # Run the detrending.
    flux_detrend, trend = wotan.flatten(time,
                                        flux,
                                        method=method,
                                        window_length=window_length,
                                        return_trend=True,
                                        **wotan_kwargs)

    # Adjust the errors.
    flux_detrend_err = flux_err/trend

    # Remove any rejected edges.
    mask = np.isfinite(trend)
    time = time[mask]
    flux_detrend = flux_detrend[mask]
    flux_detrend_err = flux_detrend_err[mask]
    trend = trend[mask]

    return time, flux_detrend, flux_detrend_err, trend


def _filter_ysd_lowess(time: np.ndarray,
                       flux: np.ndarray,
                       window_length: float,
                       gap_size: float = 0.2,
                       prominence: float = 0.001,
                       width: tuple[Optional[int], Optional[int]] = (20, None),
                       debug: bool = False
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Perform YSD-LOWESS on a lightcurve segment.

    Parameters
    ----------
    time: np.ndarray
        Array of observation times.
    flux: np.ndarray
        Array of flux measurements corresponding to the observation times.
    window_length: float
        The size of the smoothing window used with wotan, must have the same units as time.
    gap_size: float
        The size of the data gap to create at peaks and troughs.
    prominence: float
        The minimum prominence of the peaks and troughs.
    width: tuple
        The minimum and maximum width of the peaks and troughs.
    debug:
        If True show a plot of the detrending.

    Returns
    -------
    ysd_lowess: np.ndarray
        The ysd_lowess trend.
    mask: np.ndarray
        The input values than can be used.
    lowess: np.ndarray
        The lowess trend, if no peaks and troughs were cut.

    """

    # Do a normal lowess filter.
    print('Doing LOWESS detrending.')
    _, lowess = wotan.flatten(time,
                              flux,
                              method='lowess',
                              window_length=window_length,
                              return_trend=True,
                              edge_cutoff=0.,
                              break_tolerance=None)  # None since these breaks are handled in the outer function.

    # Detect peaks and throughs on the lowess filter.
    print('Detecting peaks and troughs on LOWESS trend.')
    peaks, peak_info = signal.find_peaks(lowess, prominence=prominence, width=width)
    troughs, trough_info = signal.find_peaks(-lowess, prominence=prominence, width=width)

    print(f"Detected {len(peaks)} peaks and {len(troughs)} throughs.")
    # print(peak_info, trough_info)

    # Build the peaks and throughs mask.
    near_peak_or_trough = np.zeros_like(flux, dtype='bool')

    for i in peaks:
        mask = np.abs(time - time[i]) < gap_size/2
        near_peak_or_trough = near_peak_or_trough | mask

    for i in troughs:
        mask = np.abs(time - time[i]) < gap_size/2
        near_peak_or_trough = near_peak_or_trough | mask

    if np.any(near_peak_or_trough):
        # Mask the peaks and throughs.
        flux_cut = flux[~near_peak_or_trough]
        time_cut = time[~near_peak_or_trough]

        # Perform YSD-lowess detrending.
        print('Doing YSD-LOWESS detrending.')
        _, ysd_lowess = wotan.flatten(time_cut,
                                      flux_cut,
                                      method='lowess',
                                      window_length=window_length,
                                      return_trend=True,
                                      edge_cutoff=0.,
                                      break_tolerance=0.95*gap_size)  # TODO Not sure this is safe.

        # Interpolate the YSD-LOWESS trend over the peaks and troughs.
        poly_function = interpolate.interp1d(time_cut, ysd_lowess, kind='cubic', bounds_error=False, fill_value=np.nan)
        ysd_lowess = poly_function(time)
        mask = ~np.isnan(ysd_lowess)
        ysd_lowess[~mask] = 1.
    else:
        ysd_lowess = np.copy(lowess)
        mask = np.ones_like(ysd_lowess, dtype='bool')

    if debug:
        plt.figure(figsize=(13, 5))

        # Plot the lightcurve segment.
        plt.plot(time[~near_peak_or_trough], flux[~near_peak_or_trough], '.')
        plt.plot(time[near_peak_or_trough], flux[near_peak_or_trough], '.')

        # Plot the LOWESS and YSD-LOWESS trends.
        plt.plot(time, lowess, lw=2, label='wotan LOWESS')
        plt.plot(time, ysd_lowess, lw=2, label='wotan YSD-LOWESS')

        # Mark the peaks and troughs.
        ymax = lowess[peaks]
        ymin = ymax - peak_info['prominences']
        plt.vlines(x=time[peaks], ymin=ymin, ymax=ymax, color='k', ls='-', lw=2)

        ymax = lowess[troughs]
        ymin = ymax + trough_info['prominences']
        plt.vlines(x=time[troughs], ymin=ymin, ymax=ymax, color='k', ls='--', lw=2)

        plt.xlabel('Time [days]')
        plt.ylabel('Flux')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

    return ysd_lowess, mask, lowess


def filter_ysd_lowess(time: np.ndarray,
                      flux: np.ndarray,
                      flux_err: np.ndarray,
                      window_length: float,
                      cadence: float,
                      gap_size: float = 0.2,
                      min_width: float = 7.5 / 24,
                      max_width: Optional[float] = None,
                      prominence: float = 0.001,
                      debug: bool = False
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Perform YSD-LOWESS on a lightcurve segment.

    Parameters
    ----------
    time: np.ndarray
       Array of observation times.
    flux: np.ndarray
        Array of flux measurements corresponding to the observation times.
    flux_err: np.ndarray
        Array of flux errors corresponding to the observation times.
    window_length: float
        The size of the smoothing window used with wotan, same units as time.
    cadence: float
        Cadence of the observations, same units as time.
    gap_size: float
        The size of the data gap to create at peaks and troughs, same units as time.
    min_width: float
        The minimum width of the peaks and troughs, same units as time.
    max_width: float or None
        The maximum width of the peaks and troughs, same units as time. By
        default max_width = 2 x window_length.
    prominence: float
        The minimum prominence of the peaks and troughs.
    debug: bool
        If True show a plot of the detrending.

    Returns
    -------
    time: np.ndarray
        Array of observation times, may be shorter than the input.
    flux_detrend: np.ndarray
        Detrended flux values corresponding to time.
    flux_detrend_error: np.ndarray
        Detrended error values corresponding to time.
    trend: np.ndarray
        The trend that was removed from the flux and error values.

    """

    if max_width is None:
        max_width = 2 * window_length

    min_width = np.floor(min_width / cadence)
    max_width = np.ceil(max_width / cadence)

    # Get the indexes of the gaps.
    wotan_kwargs = get_wotan_kwargs('ysd-lowess')
    gaps_indexes = wotan.gaps.get_gaps_indexes(time, break_tolerance=wotan_kwargs['break_tolerance'])

    # Iterate over all segments.
    mask = np.ones_like(flux, dtype='bool')
    trend = np.zeros_like(flux)
    for i in range(len(gaps_indexes) - 1):

        # Min/max indices of the segment.
        imin = gaps_indexes[i]
        imax = gaps_indexes[i + 1]

        # Select data segments.
        time_view = time[imin:imax]
        flux_view = flux[imin:imax]

        # Perform the filtering.
        median_val = np.median(flux_view)
        result = _filter_ysd_lowess(time_view,
                                    flux_view / median_val,
                                    window_length,
                                    gap_size=gap_size,
                                    prominence=prominence,
                                    width=(min_width, max_width),
                                    debug=debug)
        filter_segment, mask_segment, _ = result

        # Update the filtering results.
        mask[imin:imax] = mask_segment
        trend[imin:imax] = filter_segment * median_val

    # Compute the filtered flux and error.
    flux_detrend = flux / trend
    flux_detrend_err = flux_err / trend

    # Remove masked points.
    time = time[mask]
    flux_detrend = flux_detrend[mask]
    flux_detrend_err = flux_detrend_err[mask]
    trend = trend[mask]

    return time, flux_detrend, flux_detrend_err, trend


def main():
    return


if __name__ == '__main__':
    main()
