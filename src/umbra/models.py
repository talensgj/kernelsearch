import logging
from typing import Union, Optional, get_args

import numpy as np
from numpy.typing import ArrayLike

import batman

from . import utils

#############
#  LOGGING  #
#############

logger = logging.getLogger(__name__)

LOGDEBUG = logger.debug
LOGINFO = logger.info
LOGWARNING = logger.warning
LOGERROR = logger.error
LOGEXCEPTION = logger.exception


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


def get_period_kepler(stellar_density: Union[float, np.ndarray],
                      sm_axis: Union[float, np.ndarray]
                      ) -> Union[float, np.ndarray]:
    """ Compute the orbital period using Kepler's 3rd law.

    Parameters
    ----------
    stellar_density: float
        The density of the host star in g/cm^3.
    sm_axis: float or np.ndarray
        The semi-major axis in units of stellar radii.

    Returns
    -------
    period: float or np.ndarray
        The orbital period in days.

    """

    stellar_density *= 1e3  # kg/m^3

    factor = stellar_density / sm_axis ** 3
    period_s = np.sqrt(3 * np.pi / (utils.GRAVITY * factor))

    period = period_s / utils.SEC_IN_DAY

    return period


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

    ld_pars = utils._verify_ld_params(ld_type, ld_pars)

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


def _warped_lstsq_init(mid_times: np.ndarray,
                       exp_cadence: float,
                       smooth_window: float,
                       smooth_weights: utils.SmoothWeights
                       ) -> tuple[np.ndarray, np.ndarray]:
    """ Prepare the special time and weights arrays for computing WLS templates.

    Parameters
    ----------
    mid_times: np.ndarray
        The times at which to evaluate the transit model.
    exp_cadence: float
        The exposure cadence of the observations in days.
    smooth_window: float
        The smoothing window to use when generating WLS templates, should match
        whatever filter was applied to the data.
    smooth_weights: str
        The weights to apply across the smoothing window when generating WLS
        templates, should match whatever filter was applied to the data.

    Returns
    -------
    wls_times: np.ndarray
        The times at which to evaluate the transit model in order to generate
        warped transit shapes.
    wls_weights: np.ndarray
        The weights to use when generating warped transit shapes.

    """

    # Create the grid of exposures inside the smoothing window.
    nevals = np.ceil(smooth_window / exp_cadence).astype('int')
    if nevals % 2 == 0:
        nevals += 1

    mid_idx = nevals // 2
    dt = (np.arange(nevals) - mid_idx) * exp_cadence

    # Compute the weights across the smoothing window.
    if smooth_weights == 'uniform':
        weights = np.ones_like(dt)

    if smooth_weights == 'tricube':
        radius = smooth_window / 2
        weights = np.where(np.abs(dt) < radius, (1 - np.abs(dt / radius) ** 3) ** 3, 0)

    # Normalise the weights.
    weights = weights / np.sum(weights)

    # Get the final arrays of transit times and weights to compute warped transits.
    wls_times = dt[:, np.newaxis] + mid_times[np.newaxis, :]
    wls_weights = weights[:, np.newaxis]

    return wls_times, wls_weights


def _lstsq_templates(mid_times: np.ndarray,
                     duration_grid: np.ndarray,
                     transit_params: dict,
                     supersample_factor: int,
                     ld_type: utils.LDType,
                     ld_pars: ArrayLike,
                     exp_time: float,
                     exp_cadence: float,
                     smooth_window: Optional[float],
                     smooth_weights: utils.SmoothWeights
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Compute the transit shapes needed to generate the least-squares templates.

    Parameters
    ----------
    mid_times: np.ndarray
        The times at which to evaluate the transit model.
    duration_grid: np.ndarray
        The durations for which to generate transits.
    transit_params: dict
        The transit parameters to use for the transit models. The semi-major
        axis will be adjusted to match each duration.
    supersample_factor: int
        Value passed to batman for integrating longer exposures.
    ld_type: str
        The limb-darkening law to use for the TLS or WLS templates. Can be any
        law valid in the batman package.
    ld_pars: array-like
        The limb-darkening parameters to use.
    exp_time: float
        The exposure time of the observations in days.
    exp_cadence: float
        The exposure cadence of the observations in days.
    smooth_window: float or None
        The smoothing window to use when generating WLS templates, should match
        whatever filter was applied to the data.
    smooth_weights: str
        The weights to apply across the smoothing window when generating WLS
        templates, should match whatever filter was applied to the data.

    """

    # Generate the time and weights arrays for WLS.
    if smooth_window is not None:
        result = _warped_lstsq_init(mid_times, exp_cadence, smooth_window, smooth_weights)
        wls_times, wls_weights = result

        # Need 1D time array for batman.
        wls_times_shape = wls_times.shape
        wls_times = wls_times.ravel()

    # Create output arrays.
    nrows = len(duration_grid)
    ncols = len(mid_times)
    bls_template = np.zeros((nrows, ncols))
    tls_template = np.zeros((nrows, ncols))
    wls_template = np.zeros((nrows, ncols))

    # Iterate over the transit durations.
    for row_idx, transit_duration in enumerate(duration_grid):

        # Compute the scaled semi-major axis that gives the required duration.
        sm_axis = get_sm_axis(transit_params['P'],
                              transit_duration,
                              transit_params['R_p/R_s'],
                              transit_params['b'],
                              transit_params['ecc'],
                              transit_params['w'])
        transit_params['a/R_s'] = sm_axis

        # Evaluate the boxy transit model.
        result = analytic_transit_model(mid_times,
                                        transit_params,
                                        'uniform',
                                        [],
                                        exp_time=exp_time,
                                        supersample_factor=supersample_factor,
                                        max_err=1.)
        bls_template[row_idx] = result[0]

        # Evaluate the transit shape.
        result = analytic_transit_model(mid_times,
                                        transit_params,
                                        ld_type,
                                        ld_pars,
                                        exp_time=exp_time,
                                        supersample_factor=supersample_factor,
                                        max_err=1.)
        fac = result[5]  # Save fac for WLS templates.
        tls_template[row_idx] = result[0]

        if smooth_window is not None:
            # Evaluate the transit model.
            result = analytic_transit_model(wls_times,
                                            transit_params,
                                            ld_type,
                                            ld_pars,
                                            exp_time=exp_time,
                                            supersample_factor=supersample_factor,
                                            fac=fac,
                                            max_err=1.)

            wls_flux = result[0]
            wls_flux = wls_flux.reshape(wls_times_shape)
            wls_template[row_idx] = tls_template[row_idx] / np.sum(wls_weights * wls_flux, axis=0)

    return bls_template, tls_template, wls_template


def get_lstsq_templates(periods: np.ndarray,
                        duration_grid: np.ndarray,
                        epoch_step: float,
                        exp_time: float,
                        exp_cadence: float,
                        ld_type: utils.LDType = 'linear',
                        ld_pars: ArrayLike = (0.6,),
                        ref_depth: float = 5000,
                        ref_impact: float = 0.,
                        search_mode: utils.SearchMode = 'TLS',
                        smooth_window: Optional[float] = None,
                        smooth_weights: utils.SmoothWeights = 'uniform'
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Get the least-squares transit templates for the chosen search mode.

    Parameters
    ----------
    periods: np.ndarray
        The orbital periods for which we are computing the templates.
    duration_grid: np.ndarray
        The durations for which we are computing the templates.
    epoch_step: float
        The chosen size of the epoch step.
    exp_time: float
        The exposure time of the observations in days.
    exp_cadence: float
        The exposure cadence of the observations in days.
    ld_type: str
        The limb-darkening law to use for the TLS or WLS templates. Can be any
        law valid in the batman package (default: 'linear').
    ld_pars: array-like
        The limb-darkening parameters to use (default: (0.6,)).
    ref_depth: float
        The transit depth (Rp/Rs)^2 to use when making the templates in ppm
        (default: 5000 ppm).
    ref_impact: float
        The impact parameter to use when making the templates (default: 0).
    search_mode: str
        The type of transit templates to use, can be 'BLS', 'TLS' or 'WLS'
        (default: 'TLS').
    smooth_window: float or None
        The smoothing window to use when search_mode = 'WLS', should match any
        whatever filter was applied to the data (default: None).
    smooth_weights: str
        The weights to apply across the smoothing window when search_mode = 'WLS',
        should match whatever filter was applied to the data and can be 'uniform'
        or 'tricube' (default: 'uniform').

    Returns
    -------
    template_edges: np.ndarray
        The edges of the time bins in which the transit shape was computed.
    template_model: np.ndarray
        The scaled transit models for each duration value.
    template_square: np.ndarray
        The square of template_models, pre-computed for the transit search.
    template_count: np.ndarray
        Has value 1 in transit and 0 outside, used to count in-transit points
        during the transit search.

    """

    utils._verify_observation_params(exp_time, exp_cadence)
    ld_pars = utils._verify_ld_params(ld_type, ld_pars)
    smooth_window = utils._verify_lstsq_params(search_mode, smooth_window, smooth_weights, 'TLS')

    # Convert depth from ppm to fraction.
    ref_depth = 1e-6 * ref_depth

    # Get extreme values.
    min_period = np.amin(periods)
    max_period = np.amax(periods)
    max_duration = np.amax(duration_grid)

    # Check the baseline.
    baseline = min_period - max_duration - exp_time
    if search_mode == 'WLS' and periods.size > 1 and baseline < smooth_window:
        LOGWARNING("Cannot make WLS templates for this period range, defaulting to TLS templates.")
        search_mode: utils.SearchMode = 'TLS'

    # Compute the duration of the signal, accounting for exp_time and smooth_window.
    if search_mode in ['BLS', 'TLS']:
        delta_time = max_duration + exp_time
    else:
        delta_time = max_duration + exp_time + smooth_window

    if periods.size == 1:
        delta_time = np.minimum(delta_time, max_period)

    # Determine the times at which to evaluate the template.
    nbins = np.ceil(delta_time / epoch_step).astype('int')
    template_edges = np.linspace(-delta_time / 2, delta_time / 2, nbins + 1)
    mid_times = (template_edges[:-1] + template_edges[1:]) / 2

    # Set up the transit parameters.
    transit_params = dict()
    transit_params['T_0'] = 0.
    transit_params['P'] = max_period
    transit_params['R_p/R_s'] = np.sqrt(ref_depth)
    transit_params['a/R_s'] = 0.
    transit_params['b'] = ref_impact
    transit_params['ecc'] = 0.
    transit_params['w'] = 90.
    transit_params['Omega'] = 0.

    supersample_factor = np.ceil(exp_time * utils.SEC_IN_DAY / 10.).astype('int')

    # Compute the transit templates.
    result = _lstsq_templates(mid_times,
                              duration_grid,
                              transit_params,
                              supersample_factor,
                              ld_type,
                              ld_pars,
                              exp_time,
                              exp_cadence,
                              smooth_window,
                              smooth_weights)
    bls_template, tls_template, wls_template = result

    # Choose the final template based on the search mode.
    template_models = None
    if search_mode == 'BLS':
        template_models = (bls_template - 1) / ref_depth
    if search_mode == 'TLS':
        template_models = (tls_template - 1) / ref_depth
    if search_mode == 'WLS':
        template_models = (wls_template - 1) / ref_depth

    template_square = template_models ** 2
    template_count = (bls_template - 1) < 0

    return template_edges, template_models, template_square, template_count


def evaluate_lstsq_template(time: np.ndarray,
                            period: float,
                            midpoint: float,
                            depth: float,
                            flux_level: float,
                            template_edges: np.ndarray,
                            template_model: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """ Evaluate a transit model from a least-squares template.

    Parameters
    ----------
    time: np.ndarray
        The times at which to evaluate the least-squares template.
    period: float
        The orbital period of the transit.
    midpoint: float
        The mid-transit time of the transit.
    depth: float
        The transit depth (Rp/Rs)^2 of the transit.
    flux_level: float
        The out-of-transit flux level of the transit.
    template_edges: np.ndarray
        The edges of the time bins in which the template was computed.
    template_model: np.ndarray
        The template model corresponding to the duration of the transit.

    Returns
    -------
    phase: np.ndarray
        The phase of the observations, with the transit centered at 0.5.
    model: np.ndarray
        The transit model evaluated from the least-squares template.

    """

    phase = np.mod((time - midpoint) / period - 0.5, 1)  # Phase with transit at 0.5
    bin_idx = np.searchsorted(template_edges / period + 0.5, phase)  # Phase centered at 0.5
    template_model = np.append(np.append(0, template_model), 0)
    model = depth * template_model[bin_idx] + flux_level

    return phase, model


def main():
    return


if __name__ == "__main__":
    main()
