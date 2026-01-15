import logging
from typing import Optional

import numpy as np
from astropy import constants


#############
#  LOGGING  #
#############

logger = logging.getLogger(__name__)

LOGDEBUG = logger.debug
LOGINFO = logger.info
LOGWARNING = logger.warning
LOGERROR = logger.error
LOGEXCEPTION = logger.exception


###########
# GLOBALS #
###########

GRAVITY = constants.G.value  # m^3 kg^-1 s^-2
SECINDAY = 24. * 60. * 60.  # s/day
MIN_PERIOD = 0.1  # days


###############
# PERIOD GRID #
###############

def get_min_period(stellar_density: float, min_separation: float = 3.) -> float:
    """ Compute the shortest period to be searched based on the planet-star
        separation on a circular orbit.

    Parameters
    ----------
    stellar_density: float
        The density of the host star in g/cm^3.
    min_separation: float
        The minimum orbital separation between the host star and the planet in
        stellar radii (default: 3).

    Returns
    -------
    min_period: float
        The shortest period at which the separation requirement is met in days.

    """

    stellar_density = stellar_density * 1e3  # kg/m^3

    min_period = np.sqrt(min_separation ** 3 * 3 * np.pi / GRAVITY / stellar_density)  # seconds
    min_period /= SECINDAY  # days

    return min_period


def get_max_period(baseline: float, min_transits: int = 3) -> float:
    """ Compute the longest period to be searched based on the number of
        transits.

    Parameters
    ----------
    baseline: float
        The time between the start and end of observations in days.
    min_transits: int
        The minimum number of transits observed between the start and end of
        observations (default: 3).

    Returns
    -------
    max_period: float
        The longest orbital period at which the required number of transits
        could have been observed in days.

    """

    max_period = baseline / min_transits

    return max_period


def _period_grid_constants(stellar_density: float,
                           min_freq: float,
                           baseline: float,
                           oversampling: int
                           ) -> tuple[float, float]:
    """ Compute the A and C constants of Ofir (2014).
    """

    density_star = stellar_density * 1e3  # kg/m^3

    a_cubed = 3 / np.pi ** 2 * 1 / (GRAVITY * density_star)
    a_cubed /= SECINDAY ** 2

    A = a_cubed ** (1/3) / (baseline * oversampling)
    C = min_freq ** (1/3) - A/3

    return A, C


def get_period_grid(max_stellar_density: float,
                    baseline: float,
                    min_period: Optional[float] = None,
                    max_period: Optional[float] = None,
                    oversampling: int = 3,
                    min_transits: int = 3,
                    min_separation: float = 3.
                    ) -> np.ndarray:
    """ Compute the period grid to search, using the prescription of Ofir (2014).

    Parameters
    ----------
    max_stellar_density: float
        The maximum density of the host star in g/cm^3.
    baseline: float
        The time between the start and end of observations in days.
    min_period: float, optional
        The shortest period to search in days (default: None).
    max_period: float, optional
        The longest period to search in days (default: None).
    oversampling: int
        The oversampling factor of the period grid (default: 3).
    min_transits: int
        The minimum number of transits observed between the start and end of
        observations, used to set max_period if not provided (default: 3).
    min_separation: float
        The minimum orbital separation between the host star and the planet in
        stellar radii, used to set min_period if not provided (default: 3).

    Returns
    ------
    period_grid: np.ndarray
        Array of period values to search.

    """

    if min_period is None:
        min_period = get_min_period(max_stellar_density, min_separation)

    if max_period is None:
        max_period = get_max_period(baseline, min_transits)

    if min_period < MIN_PERIOD:
        min_period = get_min_period(max_stellar_density, min_separation)
        LOGWARNING(f"Provided minimum period is less than {MIN_PERIOD} days, using {min_period} days instead.")

    if max_period > baseline:
        max_period = baseline
        LOGWARNING(f"Provided maximum period exceeds the baseline, using the {max_period} days baseline instead.")

    if max_period < min_period:
        msg = f"The maximum period is less than the minimum period, please fix your inputs."
        raise ValueError(msg)

    # Compute the minimum and maximum frequency.
    min_freq = 1 / max_period
    max_freq = 1 / min_period

    # Compute the A and C values from Ofir (2014).
    A, C = _period_grid_constants(max_stellar_density, min_freq, baseline, oversampling)

    # Compute the number of samples.
    max_xval = 3/A * (max_freq ** (1/3) - C)
    max_xval = np.ceil(max_xval).astype('int')

    # Compute the period grid.
    xvals = np.arange(max_xval) + 1
    freq_grid = (A/3 * xvals + C) ** 3
    period_grid = 1 / freq_grid[::-1]

    return period_grid


def main():
    return


if __name__ == "__main__":
    main()
