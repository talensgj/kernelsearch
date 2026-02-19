from typing import Union, Optional, get_args

import numpy as np
from numpy.typing import ArrayLike

import batman

from . import utils

RNG = np.random.default_rng(5627323756)


def _ecc_factors(eccentricity: Union[float, np.ndarray],
                 arg_periastron: Union[float, np.ndarray]
                 ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """ Compute 2 eccentricity terms that appear in other equations.
    """

    arg_periastron = np.deg2rad(arg_periastron)

    x = 1 - eccentricity ** 2
    y = 1 + eccentricity * np.sin(arg_periastron)

    alpha = np.sqrt(x) / y
    beta = x / y

    return alpha, beta


def get_orbital_inclination(impact_param: Union[float, np.ndarray],
                            sm_axis: Union[float, np.ndarray],
                            eccentricity: Union[float, np.ndarray],
                            arg_periastron: Union[float, np.ndarray]
                            ) -> Union[float, np.ndarray]:
    """ Compute the orbital inclination (i) for an eccentric orbit, using
        Equation 7 from Winn (2010).

    Parameters
    ----------
    impact_param: float or np.ndarray
        The impact parameter value(s).
    sm_axis: float or np.ndarray
        The semi-major axis value(s) in stellar radii.
    eccentricity: float or np.ndarray
        The orbital eccentricty value(s).
    arg_periastron: float or np.ndarray
        The argument of periastron in degrees.

    Returns
    -------
    inclination: float or np.ndarray
        The orbital inclination in degrees.

    """

    alpha, beta = _ecc_factors(eccentricity, arg_periastron)
    inclination = np.arccos(impact_param / (sm_axis * beta))
    inclination = np.rad2deg(inclination)

    return inclination


def get_impact_parameter(inclination: Union[float, np.ndarray],
                         sm_axis: Union[float, np.ndarray],
                         eccentricity: Union[float, np.ndarray],
                         arg_periastron: Union[float, np.ndarray]
                         ) -> Union[float, np.ndarray]:
    """ Compute the impact parameter (b) for an eccentric orbit, using
        Equation 7 from Winn (2010).

    Parameters
    ----------
    inclination: float or np.ndarray
        The orbital inclination values(s) in degrees.
    sm_axis: float or np.ndarray
        The semi-major axis value(s) in stellar radii.
    eccentricity: float or np.ndarray
        The orbital eccentricty value(s).
    arg_periastron: float or np.ndarray
        The argument of periastron values(s) in degrees.

    Returns
    -------
    impact_param: float or np.ndarray
        The impact parameter value(s).

    """

    inclination = np.deg2rad(inclination)
    alpha, beta = _ecc_factors(eccentricity, arg_periastron)
    impact_param = sm_axis * beta * np.cos(inclination)

    return impact_param


def get_transit_duration(period: Union[float, np.ndarray],
                         sm_axis: Union[float, np.ndarray],
                         planet_radius: Union[float, np.ndarray],
                         impact_param: Union[float, np.ndarray],
                         eccentricity: Union[float, np.ndarray],
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
    eccentricity: float or np.ndarray
        The orbital eccentricty value(s).
    arg_periastron: float or np.ndarray
        The argument of periastron value(s) in degrees.

    Returns
    -------
    transit_duration: float or np.ndarray
        The transit duration value(s) in days.

    """

    alpha, beta = _ecc_factors(eccentricity, arg_periastron)
    sin_sq = beta ** 2 * ((1 + planet_radius) ** 2 - impact_param ** 2) / (beta ** 2 * sm_axis ** 2 - impact_param ** 2)
    transit_duration = alpha * period / np.pi * np.arcsin(np.sqrt(sin_sq))

    return transit_duration


def get_sm_axis(period: Union[float, np.ndarray],
                transit_duration: Union[float, np.ndarray],
                planet_radius: Union[float, np.ndarray],
                impact_param: Union[float, np.ndarray],
                eccentricity: Union[float, np.ndarray],
                arg_periastron: Union[float, np.ndarray]
                ) -> Union[float, np.ndarray]:
    """ Compute the semi-major axis that produces a sppecific duration for an
        eccentric orbit, using Equations 7, 14 and 16 from Winn (2010).
    """

    alpha, beta = _ecc_factors(eccentricity, arg_periastron)
    sin_sq = np.sin(transit_duration / alpha * np.pi / period) ** 2
    sm_axis_sq = ((1 + planet_radius) ** 2 - impact_param ** 2)/sin_sq + impact_param ** 2 / beta ** 2
    sm_axis = np.sqrt(sm_axis_sq)

    return sm_axis


def get_stellar_density_kepler(sm_axis: Union[float, np.ndarray],
                               period: Union[float, np.ndarray]
                               ) -> Union[float, np.ndarray]:
    """ Compute the stellar density using Kepler's 3rd law.

    Parameters
    ----------
    sm_axis: float or np.ndarray
        The semi-major axis in units of stellar radii.
    period: float or np.ndarray
        The orbital period in days.

    Returns
    -------
    stellar_density: float
        The density of the host star in g/cm^3.

    """

    period_s = period * utils.SEC_IN_DAY  # seconds

    factor = 3 * np.pi / (utils.GRAVITY * period_s ** 2)
    stellar_density = factor * sm_axis ** 3

    stellar_density /= 1e3  # g/cm^3

    return stellar_density


def get_sm_axis_kepler(stellar_density: Union[float, np.ndarray],
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
    sm_axis: float or np.ndarray
        The semi-major axis in units of stellar radii.

    """

    stellar_density *= 1e3  # kg/m^3
    period_s = period * utils.SEC_IN_DAY  # seconds

    factor = 3 * np.pi / (utils.GRAVITY * period_s ** 2)
    sm_axis = (stellar_density / factor) ** (1 / 3)

    return sm_axis


def analytic_transit_model(time: np.ndarray,
                           transit_params: dict,
                           ld_type: utils.LDType,
                           ld_pars: ArrayLike,
                           exp_time: Optional[float] = None,
                           supersample_factor: Optional[int] = None,
                           fac: Optional[float] = None,
                           max_err: float = 0.5,
                           return_orbit: bool = False
                           ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], batman.TransitParams, float]:
    # TODO stricter max_err avoids convergence issue?
    """ Evaluate the transit model using batman.

    Parameters
    ----------
    time: np.ndarray
        Array of times at which to compute the lightcurve.
    transit_params: dict
        Dictionary contaning the orbital parameters.
    ld_type: str
        The limb-darkening law to use for the star.
    ld_pars: array-like
        The coefficients of the limb-darkening polynomial.
    exp_time: float or None
        The exposure time to use when evaluating the transit model.
    supersample_factor: int or None
        The supersampling to use when evaluating with an exposure time set.
    fac: float or none
        The batman.TransitModel fac parameter.
    max_err: float
        The batman.TransitModel max_err parameter.
    return_orbit: bool
        If True the nu, xp, yp arrays are computed and returned.

    Returns
    -------
    flux: np.ndarray
        The relative flux of the star.
    nu: np.ndarray or None
        The true anomaly of the orbit, returned only if return_orbit=True.
    xp: np.ndarray or None
        The x-coordinate of the planet orbit, returned only if return_orbit=True.
    yp: np.ndarray or None
        The y-coordinate of the planet orbit, returned only if return_orbit=True.
    params: batman.TransitParams
        The TransitParams instance used.
    fac: float or None
        The batman.TransitModel fac parameter used.

    """

    if ld_type not in get_args(utils.LDType):
        msg = f"Invalid value '{ld_type}' for parameter ld_type."
        raise ValueError(msg)

    if exp_time is None:
        exp_time = 0.
        supersample_factor = 1

    # Input parameters.
    t0 = transit_params['T_0']
    per = transit_params['P']
    rp = transit_params['R_p/R_s']
    a = transit_params['a/R_s']
    b = transit_params['b']
    ecc = transit_params['ecc']
    w = transit_params['w']
    Omega = transit_params['Omega']

    # Derived parameters.
    inc = get_orbital_inclination(b, a, ecc, w)

    # Create an instance of the batman transit model.
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w

    params.u = ld_pars
    params.limb_dark = ld_type

    # Create the TransitModel instance.
    model = batman.TransitModel(params, time, fac=fac, max_err=max_err, exp_time=exp_time, supersample_factor=supersample_factor)

    # Compute the true anomaly and the flux.
    flux = model.light_curve(params)

    nu, xp, yp = None, None, None
    if return_orbit:

        # Convert angles to radians.
        inc = np.deg2rad(inc)
        w = np.deg2rad(w)
        Omega = np.deg2rad(Omega)

        # Compute the planets orbit in the plane of the sky.
        nu = model.get_true_anomaly()
        r = (1 - ecc ** 2) / (1 + ecc * np.cos(nu))
        xi = a * r * np.sin(nu + w - np.pi / 2.)
        yi = a * r * np.cos(nu + w - np.pi / 2.) * np.cos(inc)
        xp = xi * np.cos(Omega) - yi * np.sin(Omega)
        yp = xi * np.sin(Omega) + yi * np.cos(Omega)

    return flux, nu, xp, yp, params, model.fac
