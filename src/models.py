from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

import batman
from astropy import constants, units

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


def axis2full(a, per, p, b, ecc, w):
    """"""

    sin_sq = ((1 - p) ** 2 - b ** 2) / (a ** 2 - b ** 2)
    transit_full = per / np.pi * np.arcsin(np.sqrt(sin_sq))

    # Eccentricity correction (for transits).
    transit_full = transit_full * np.sqrt(1 - ecc ** 2) / (1 + ecc * np.sin(w * DEG2RAD))

    return transit_full


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


def transit_masks(time: np.ndarray,
                  transit_params: dict,
                  window: float = 3.
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Compute various useful masks and number from orbital parameters.
    """

    # Input parameters.
    t0 = transit_params['T_0']
    per = transit_params['P']
    rp = transit_params['R_p/R_s']
    a = transit_params['a/R_s']
    b = transit_params['b']
    ecc = transit_params['ecc']
    w = transit_params['w']

    transit_duration = axis2duration(a, per, rp, b, ecc, w)

    phase = (time - t0) / per
    norbit = np.round(phase).astype('int')

    norbit_unique = np.unique(norbit)
    midpoint = t0 + norbit_unique * per

    phase = np.mod(phase - 0.5, 1) - 0.5
    intransit = np.abs(phase) <= 0.5 * transit_duration / per
    baseline = (np.abs(phase) <= 0.5 * window * transit_duration / per) & ~intransit

    min_orbit = np.amin(norbit_unique)
    norbit = norbit - min_orbit
    norbit_unique = norbit_unique - min_orbit

    return norbit, intransit, baseline, norbit_unique, midpoint


def latlon2xy(lat, lon, lat0, lon0):
    """ Convert latitude and longitude to x, y coordinates using an orthographic
        projection.
    """

    # Convert degrees to radians.
    lat, lon = lat * DEG2RAD, lon * DEG2RAD
    lat0, lon0 = lat0 * DEG2RAD, lon0 * DEG2RAD

    xvec = np.cos(lat) * np.sin(lon - lon0)
    yvec = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    visible = cos_c >= 0

    return xvec, yvec, visible


def xy2latlon(x, y, lat0, lon0):
    """ Convert x, y coordinates to latitude and longitude using an orthographic
        projection.
    """

    lat0, lon0 = lat0 * DEG2RAD, lon0 * DEG2RAD

    rho = np.sqrt(x ** 2 + y ** 2)
    c = np.arcsin(rho)

    tmp1 = np.cos(c) * np.sin(lat0) + y * np.cos(lat0)
    lat = np.arcsin(tmp1)
    tmp2 = rho * np.cos(c) * np.cos(lat0) - y * np.sin(c) * np.sin(lat0)
    lon = lon0 + np.arctan2(x * np.sin(c), tmp2)

    lat, lon = lat * RAD2DEG, lon * RAD2DEG

    return lat, lon


def analytic_transit_model(time: np.ndarray,
                           transit_params: dict,
                           ld_type: str,
                           ld_pars: ArrayLike,
                           exp_time: Optional[float] = None,
                           supersample_factor: Optional[int] = None,
                           fac: Optional[float] = None,
                           max_err: float = 0.5
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, batman.TransitParams, float]:
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
    fac: float or none
        The batman.TransitModel fac parameter.
    max_err: float
        The batman.TransitModel max_err parameter.

    Returns
    -------
    flux: np.ndarray
        The relative flux of the star.
    nu: np.ndarray
        The true anomaly of the orbit.
    xp: np.ndarray
        The x-coordinate of the planet orbit.
    yp: np.ndarray
        The y-coordinate of the planet orbit.
    params: batman.TransitParams
        The TransitParams instance used.
    fac: float or None
        The batman.TransitModel fac parameter used.

    """

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
    nu = model.get_true_anomaly()
    flux = model.light_curve(params)

    # Convert angles to radians.
    inc = inc * DEG2RAD
    w = w * DEG2RAD
    Omega = Omega * DEG2RAD

    # Compute the planets orbit in the plane of the sky.
    r = (1 - ecc ** 2) / (1 + ecc * np.cos(nu))
    xi = a * r * np.sin(nu + w - np.pi / 2.)
    yi = a * r * np.cos(nu + w - np.pi / 2.) * np.cos(inc)
    xp = xi * np.cos(Omega) - yi * np.sin(Omega)
    yp = xi * np.sin(Omega) + yi * np.cos(Omega)

    return flux, nu, xp, yp, params, model.fac