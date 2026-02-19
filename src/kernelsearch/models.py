from typing import Optional, get_args

import numpy as np
from numpy.typing import ArrayLike

import batman
from astropy import constants, units

from . import utils

RNG = np.random.default_rng(5627323756)
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


def impact2inc(b, a, ecc, w):
    """ Convert the impact parameter to orbital inclination.
    """

    factor = a * (1 - ecc ** 2) / (1 + ecc * np.sin(w * DEG2RAD))
    inc = np.arccos(b / factor) * RAD2DEG

    return inc


def inc2impact(inc, a, ecc, w):
    """ Convert orbital inclination to the impact parameter.
    """

    factor = a * (1 - ecc ** 2) / (1 + ecc * np.sin(w * DEG2RAD))
    impact = factor * np.cos(inc * DEG2RAD)

    return impact


def axis2duration(a, per, p, b, ecc, w):
    """ Convert the scaled semi-major axis to transit duration.
    """

    # Duration in the case of a circular orbit.
    sin_sq = ((1 + p) ** 2 - b ** 2) / (a ** 2 - b ** 2)
    transit_duration = per / np.pi * np.arcsin(np.sqrt(sin_sq))

    # Eccentricity correction (for transits).
    transit_duration = transit_duration * np.sqrt(1 - ecc ** 2) / (1 + ecc * np.sin(w * DEG2RAD))

    return transit_duration


def duration2axis(transit_duration, per, p, b, ecc, w):
    """ Convert the scaled semi-major axis to transit duration.
    """

    # Eccentricity correction (for transits).
    transit_duration = transit_duration / (np.sqrt(1 - ecc ** 2) / (1 + ecc * np.sin(w * DEG2RAD)))

    # Duration in the case of a circular orbit.
    sin_sq = np.sin(transit_duration/per*np.pi)**2
    asq = ((1 + p) ** 2 - b ** 2)/sin_sq + b ** 2

    return np.sqrt(asq)


def axis2density(a, per):
    """ Convert the scaled semi-major axis to the stellar density in cgs units.
    """

    per = per * units.day

    factor = 3 * np.pi / (constants.G * per ** 2)
    rho = factor * a ** 3

    rho = rho.to(units.g / units.cm ** 3)

    return rho.value


def density2axis(rho, per):
    """ Convert the stellar density (in cgs) to the scaled smei-major axis.
    """

    rho = rho * units.g / units.cm ** 3
    per = per * units.day

    factor = 3 * np.pi / (constants.G * per ** 2)
    a = (rho / factor) ** (1 / 3)
    a = a.decompose()

    return a.value


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
    inc = impact2inc(b, a, ecc, w)

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
        inc = inc * DEG2RAD
        w = w * DEG2RAD
        Omega = Omega * DEG2RAD

        # Compute the planets orbit in the plane of the sky.
        nu = model.get_true_anomaly()
        r = (1 - ecc ** 2) / (1 + ecc * np.cos(nu))
        xi = a * r * np.sin(nu + w - np.pi / 2.)
        yi = a * r * np.cos(nu + w - np.pi / 2.) * np.cos(inc)
        xp = xi * np.cos(Omega) - yi * np.sin(Omega)
        yp = xi * np.sin(Omega) + yi * np.cos(Omega)

    return flux, nu, xp, yp, params, model.fac
