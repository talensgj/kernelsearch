import logging
from typing import Union, Optional
from collections import namedtuple

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

    stellar_density *= 1e3  # kg/m^3

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

    stellar_density *= 1e3  # kg/m^3

    a_cubed = 3 / np.pi ** 2 * 1 / (GRAVITY * stellar_density)
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
    max_xval = np.floor(max_xval).astype('int')

    # Compute the period grid.
    xvals = np.arange(max_xval) + 1
    freq_grid = (A/3 * xvals + C) ** 3
    period_grid = 1 / freq_grid[::-1]

    return period_grid


#################
# DURATION GRID #
#################

StableOrbit = namedtuple('stable_orbit', ['sm_axis', 'ecc'])
DurationLimits = namedtuple('duration_limits', ['short', 'long'])


def get_sm_axis(stellar_density: float,
                period: Union[float, np.ndarray]
                ) -> Union[float, np.ndarray]:
    """ Compute the scaled semi-major axis using Kepler's 3rd law.

    Parameters
    ----------
    stellar_density: float
        The density of the host star in g/cm^3.
    period: float or np.ndarray
        The orbital period in days.

    Returns
    -------
    axis: float or np.ndarray
        The semi-major axis in units of stellar radii.

    """

    stellar_density *= 1e3  # kg/m^3
    period_s = period * SECINDAY  # seconds

    factor = 3 * np.pi / (GRAVITY * period_s ** 2)
    sm_axis = (stellar_density / factor) ** (1 / 3)

    return sm_axis


def get_stable_orbits(sm_axis: np.ndarray,
                      min_separation: float = 3.
                      ) -> StableOrbit:
    """ Compute the limiting values for the semi-major axis and eccentricty
        that still result in stable orbits.

    Parameters
    ----------
    sm_axis: np.ndarray
        Semi-major axis values in units of stellar radii.
    min_separation: float
        The minimum orbital separation between the host star and the planet in
        stellar radii. Orbits are stable if the separation at closest approach
        is greater than this value (default: 3).

    Returns
    -------
    stable_orbit: StableOrbit
        Named tuple containing the semi-major axis and eccentricty of the
        most eccentric stable orbit.

    """

    # Not all densities produce stable orbits at low periods.
    sm_axis_stable = np.maximum(sm_axis, min_separation)

    # Not all eccentricties produce stable orbits.
    # Stable orbits asymptotically approach ecc -> 1 as a/R -> inf.
    ecc_stable = (sm_axis_stable - min_separation)/sm_axis_stable

    stable_orbit = StableOrbit(sm_axis=sm_axis_stable, ecc=ecc_stable)

    return stable_orbit


def get_orbit_bounds(period_grid: np.ndarray,
                     min_stellar_density: float,
                     max_stellar_density: float,
                     min_separation: float = 3.
                     ) -> tuple[StableOrbit, StableOrbit]:
    """ Given an array of period values and a stellar density interval, compute
        the bounding semi-major axis and eccentricity values of the inner and
        outer stable orbits represented by the density interval.

    Parameters
    ----------
    period_grid: np.ndarray
        An array of orbital periods in days.
    min_stellar_density: float
        The lower bound on the stellar density in g/cm^3.
    max_stellar_density: float
        The upper bound on the stellar density in g/cm^3.
    min_separation: float
        The minimum orbital separation between the host star and the planet in
        stellar radii. Orbits are stable if the separation at closest approach
        is greater than this value (default: 3).

    Returns
    -------
    inner_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        innermost stable orbit.
    outer_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        outermost stable orbit.

    """

    # Compute semi-major axis values from the stellar density bounds.
    sm_axis_inner = get_sm_axis(min_stellar_density, period_grid)
    sm_axis_outer = get_sm_axis(max_stellar_density, period_grid)

    # Check the range of semi-major axes and eccentricties that produce stable orbits.
    inner_orbit = get_stable_orbits(sm_axis_inner, min_separation=min_separation)
    outer_orbit = get_stable_orbits(sm_axis_outer, min_separation=min_separation)

    return inner_orbit, outer_orbit


def get_transit_duration(period: Union[float, np.ndarray],
                         sm_axis: Union[float, np.ndarray],
                         planet_radius: Union[float, np.ndarray],
                         impact_param: Union[float, np.ndarray],
                         eccentricty: Union[float, np.ndarray],
                         arg_periastron: Union[float, np.ndarray]
                         ) -> Union[float, np.ndarray]:
    """ Compute the full transit duration (T14) for an eccentric orbit, using
        Equations 7, 14 and 16 from Winn (2010).

    Parameters
    ----------
    period: float or np.ndarray
        The orbital period value(s) in days.
    sm_axis: float or np.ndarray
        The semi-major axis value(s) in stellar radii.
    planet_radius: float or np.ndarray
        The planet radius value(s) in stellar radii.
    impact_param: float or np.ndarray
        The impact parameter value(s).
    eccentricty: float or np.ndarray
        The orbital eccentricty value(s).
    arg_periastron: float or np.ndarray
        The argument of periastron in degrees.

    Returns
    -------
    transit_duration: float or np.ndarray
        The transit duration in days.

    """

    arg_periastron = np.deg2rad(arg_periastron)

    # Compute the eccentricty terms.
    x = np.sqrt(1 - eccentricty ** 2)
    y = 1 + eccentricty * np.sin(arg_periastron)

    alpha = x/y
    beta = x**2/y

    # Compute the transit duration.
    sin_sq = beta ** 2 * ((1 + planet_radius) ** 2 - impact_param ** 2) / (beta ** 2 * sm_axis ** 2 - impact_param ** 2)
    transit_duration = alpha * period / np.pi * np.arcsin(np.sqrt(sin_sq))

    return transit_duration


def get_transit_duration_limits(period_grid: np.ndarray,
                                stellar_density_bounds: tuple[float, float],
                                stellar_radius_bounds: tuple[float, float],
                                planet_radius_bounds: tuple[float, float] = (0.01, 0.20),
                                impact_param_bounds: tuple[float, float] = (0.0, 0.9),
                                min_separation: float = 3.,
                                circular_orbits: bool = False,
                                ) -> tuple[DurationLimits, StableOrbit, StableOrbit]:
    """ Compute the minimum and maximum transit duration as a function of the
        orbital period, given possible bounds on the stellar density and
        valid stellar radii.

    Parameters
    ----------
    period_grid: np.ndarray
        An array of orbital periods in days.
    stellar_density_bounds: tuple
        The minimum and maximum density in g/cm^3.
    stellar_radius_bounds: tuple
        Representative stellar radii corresponding to the density bounds in
        solar radii.
    planet_radius_bounds: tuple
        The minumum and maximum planet radius in solar radii. Used when
        computing the shortest and longest transit duration respectively
        (default: (0.01, 0.20)).
    impact_param_bounds: tuple
        The minimum and maximum impact parameter. Used when computing the
        longest and shortest transit duration respectively (default:
        (0.0, 0.9)).
    min_separation: float
        The minimum orbital separation between the host star and the planet in
        stellar radii. Orbits are stable if the separation at closest approach
        is greater than this value (default: 3).
    circular_orbits: bool
        If True compute the shortest and longest duration on circular orbits,
        i.e. forces the eccentricty to zero (default: False).

    Returns
    -------
    duration_limits: DurationLimits
        A named tuple containing the short and long duration limits.
    inner_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        innermost stable orbit.
    outer_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        outermost stable orbit.

    """

    # Compute the minimum period where the density range produces stable orbits.
    min_stellar_density, max_stellar_density = stellar_density_bounds
    stellar_radius_mindens, stellar_radius_maxdens = stellar_radius_bounds
    min_planet_radius, max_planet_radius = planet_radius_bounds
    min_impact_param, max_impact_param = impact_param_bounds

    # Compute the period limits.
    period_min = get_min_period(max_stellar_density, min_separation)
    period_break = get_min_period(min_stellar_density, min_separation)

    if np.any(period_grid < period_min):
        raise ValueError("The shortest period in period_grid is incompatible with the stellar density bounds, please fix your inputs.")

    # The minimum stellar density goes up from Pbreak to Pmin.
    # The corresponding stellar radius should decrease.
    # Since there is no exact way to evolve this, we simply jump to the other radius extreme.
    stellar_radius_mindens = np.where(period_grid > period_break, stellar_radius_mindens, stellar_radius_maxdens)

    # At fixed period, the maximum density produces the shortest transit.
    # It follows that the small planet radius needs to be scaled by the
    # corresponding stellar radius.
    min_radius_ratio = min_planet_radius/stellar_radius_maxdens
    max_radius_ratio = max_planet_radius/stellar_radius_mindens

    # Compute the scaled semi-major axis and eccentricty limits for the density values.
    result = get_orbit_bounds(period_grid, min_stellar_density, max_stellar_density, min_separation=min_separation)
    inner_orbit, outer_orbit = result

    if circular_orbits:
        outer_orbit = outer_orbit._replace(ecc=np.zeros_like(period_grid))
        inner_orbit = inner_orbit._replace(ecc=np.zeros_like(period_grid))

    # Compute the duration limits for the given parameter bounds.
    duration_short = get_transit_duration(period_grid, outer_orbit.sm_axis, min_radius_ratio, max_impact_param, outer_orbit.ecc, 90.)
    duration_long = get_transit_duration(period_grid, inner_orbit.sm_axis, max_radius_ratio, min_impact_param, inner_orbit.ecc, 270.)

    duration_limits = DurationLimits(short=duration_short, long=duration_long)

    return duration_limits, inner_orbit, outer_orbit


def main():
    return


if __name__ == "__main__":
    main()
