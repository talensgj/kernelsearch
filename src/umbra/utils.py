import os.path
import logging
import multiprocessing as mp
from importlib.resources import files
from typing import Optional, Literal, get_args

import asdf
import numpy as np
from numpy.typing import ArrayLike
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

DEBUG = False

# Unit conversion quantities and constants.
HR_IN_DAY = 24
MIN_IN_DAY = 24*60
SEC_IN_DAY = 24*60*60
GRAVITY = constants.G.value  # m^3 kg^-1 s^-2
SOLAR_DENSITY = (constants.M_sun / (4/3 * np.pi * constants.R_sun ** 3)).to('g/cm^3').value

# Sanity check values.
MIN_SEPARATION = 1.20
MAX_DUTY_CYCLE = 0.30
IN_TRANSIT_FLOOR = 5
MIN_STELLAR_MASS = 0.01
MAX_STELLAR_MASS = 100.
MIN_STELLAR_RADIUS = 0.001
MAX_STELLAR_RADIUS = 1000.

# Literal type hints.
LDType = Literal["uniform", "linear", "quadratic", "square-root", "logarithmic", "exponential", "power2", "nonlinear"]
BinMethod = Literal["points", "window"]
SearchMode = Literal["BLS", "TLS", "WLS"]
ShortPeriods = Literal["skip", "TLS", "WLS"]
SmoothWeights = Literal["uniform", "tricube"]
Normalisation = Literal["simple", "umbra"]


def _verify_lightcurve(time: np.ndarray,
                       flux: np.ndarray,
                       flux_err: np.ndarray
                       ):
    """ Check the input lightcurve seems valid.
    """

    if time.ndim != 1 or flux.ndim != 1 or flux_err.ndim != 1:
        msg = f"Lightcurve arrays (time, flux, flux_err) must be 1-dimensional."
        raise ValueError(msg)

    if time.shape != flux.shape or time.shape != flux_err.shape:
        msg = f"Lightcurve arrays (time, flux, flux_err) must be the same shape."
        raise ValueError(msg)

    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)) or not np.all(np.isfinite(flux_err)):
        msg = f"Lightcurve arrays (time, flux, flux_err) cannot contain Infs or NaNs."
        raise ValueError(msg)

    return


def _verify_observation_params(exp_time: float,
                               exp_cadence: float
                               ):
    """ Check the the exposure time/cadence are valid.
    """

    if not (exp_time > 0):
        msg = f"Parameter exp_time must be greater than zero."
        raise ValueError(msg)

    if not (exp_cadence > 0):
        msg = f"Parameter exp_cadence must be greater than zero."
        raise ValueError(msg)

    return


def _verify_stellar_params(min_stellar_radius: float,
                           max_stellar_radius: float,
                           min_stellar_mass: float,
                           max_stellar_mass: float
                           ):
    """ Check the input stellar parameters are reasonable.
    """

    if min_stellar_radius > max_stellar_radius:
        msg = f"Parameter min_stellar_radius must be <= max_stellar_radius."
        raise ValueError(msg)

    if min_stellar_mass > max_stellar_mass:
        msg = f"Parameter min_stellar_mass must be <= max_stellar_mass."
        raise ValueError(msg)

    if not (MIN_STELLAR_RADIUS <= min_stellar_radius <= MAX_STELLAR_RADIUS):
        msg = f"Parameter min_stellar_radius must be in range [{MIN_STELLAR_RADIUS}, {MAX_STELLAR_RADIUS}]."
        raise ValueError(msg)

    if not (MIN_STELLAR_RADIUS <= max_stellar_radius <= MAX_STELLAR_RADIUS):
        msg = f"Parameter max_stellar_radius must be in range [{MIN_STELLAR_RADIUS}, {MAX_STELLAR_RADIUS}]."
        raise ValueError(msg)

    if not (MIN_STELLAR_MASS <= min_stellar_mass <= MAX_STELLAR_MASS):
        msg = f"Parameter min_stellar_mass must be in range [{MIN_STELLAR_MASS}, {MAX_STELLAR_MASS}]."
        raise ValueError(msg)

    if not (MIN_STELLAR_MASS <= max_stellar_mass <= MAX_STELLAR_MASS):
        msg = f"Parameter max_stellar_mass must be in range [{MIN_STELLAR_MASS}, {MAX_STELLAR_MASS}]."
        raise ValueError(msg)

    return


def _verify_ld_params(ld_type: LDType,
                      ld_pars: ArrayLike
                      ) -> np.ndarray:
    """ Check the input limb-darkening parameters ar valid.
    """

    if ld_type not in get_args(LDType):
        msg = f"Invalid value '{ld_type}' for parameter ld_type."
        raise ValueError(msg)

    ld_pars = np.asarray(ld_pars)
    if ld_pars.ndim > 1:
        msg = f"Limb-darkening coefficients (ld_pars) must be 0- or 1-dimensional."
        raise ValueError(msg)

    num_pars = ld_pars.size
    if ld_type in ["uniform"] and num_pars == 0:
        pass
    elif ld_type in ["linear"] and num_pars == 1:
        pass
    elif ld_type in ["quadratic", "square-root", "logarithmic", "exponential"] and ld_pars.size == 2:
        pass
    elif ld_type in ["power2"] and num_pars == 3:
        pass
    elif ld_type in ["nonlinear"] and num_pars == 4:
        pass
    else:
        msg = f"Invalid number of limb-darkening parameters {num_pars} for ld_type = '{ld_type}'."
        raise ValueError(msg)

    return ld_pars


def _verify_min_transits(min_transits: int):
    """ Check the min_transits parameter is valid.
    """

    if not (min_transits >= 1):
        msg = f"Parameter min_transits must be >= 1."
        raise ValueError(msg)

    return


def _verify_min_separation(min_separation: float):
    """ Check the min_separation parameter is valid.
    """

    if not (min_separation >= 1):
        msg = f"Parameter min_separation must be >= 1."
        ValueError(msg)

    if min_separation < MIN_SEPARATION:
        LOGWARNING(f"Orbits with min_separation < {MIN_SEPARATION} are unlikely to be stable.")

    return


def _verify_period_grid_params(min_transits: int,
                               min_separation: float,
                               period_sampling: int,
                               min_period: Optional[float] = None,
                               max_period: Optional[float] = None
                               ):
    """ Check the period grid parameters are valid.
    """

    _verify_min_transits(min_transits)
    _verify_min_separation(min_separation)

    if not (1 <= period_sampling <= 9):
        msg = f"Parameter period_sampling must be in range [1, 9]."
        raise ValueError(msg)

    if min_period is not None and max_period is not None and min_period >= max_period:
        msg = f"Parameter min_period must be < max_period."
        raise ValueError(msg)

    return


def _verify_epoch_grid_params(epoch_sampling: int,
                              min_epoch_step: float,
                              max_epoch_step: float
                              ):
    """ Check the epoch grid parameters are valid.
    """

    if not (2 <= epoch_sampling <= 100):
        msg = f"Parameter epoch_sampling must be in range [2, 100]."
        raise ValueError(msg)

    if not (0 < min_epoch_step < 1/HR_IN_DAY):
        msg = f"Parameter min_epoch_step must be in range (0, 1h]."
        raise ValueError(msg)

    if not (0 < max_epoch_step < 1/HR_IN_DAY):
        msg = f"Parameter max_epoch_step must be in range (0, 1h]."
        raise ValueError(msg)

    if min_epoch_step > max_epoch_step:
        msg = f"Parameter min_epoch_step must be <= max_epoch_step."
        raise ValueError(msg)

    return


def _verify_duration_lims_params(circular_orbits: bool,
                                 frac_eccentricity: float):
    """ Check the duration limits parameters are valid.
    """

    if not isinstance(circular_orbits, bool):
        msg = f"Parameter circular_orbits must be boolean."
        raise ValueError(msg)

    if not (0 < frac_eccentricity <= 1):
        msg = f"Parameter frac_eccentricity must be in range (0, 1]."
        raise ValueError(msg)

    return

def _verify_frac_duration_step(frac_duration_step: float):
    """ Check the frac_duration_step parameter is valid.
    """

    if not (1 < frac_duration_step <= 1.5):
        msg = f"Parameter frac_duration_step must be in range (1, 1.5]."
        raise ValueError(msg)

    return


def _verify_lstsq_params(search_mode: SearchMode,
                         smooth_window: Optional[float],
                         smooth_weights: SmoothWeights,
                         short_periods: ShortPeriods
                         ) -> Optional[float]:

    if search_mode not in get_args(SearchMode):
        msg = f"Invalid value '{search_mode}' for parameter search_mode."
        raise ValueError(msg)

    if search_mode in ["BLS", "TLS"]:

        if smooth_window is not None:
            LOGWARNING(f"Performing {search_mode} search, setting smooth_window to None.")
            smooth_window = None

        return smooth_window

    if smooth_window is None:
        msg = f"Parameter smooth_window cannot be None for 'WLS' search."
        raise ValueError(msg)

    if not (smooth_window > 0):
        msg = f"Parameter smooth_window must be greater than zero."
        raise ValueError(msg)

    if smooth_weights not in get_args(SmoothWeights):
        msg = f"Invalid value '{smooth_weights}' for parameter smooth_weights."
        raise ValueError(msg)

    if short_periods not in get_args(ShortPeriods):
        msg = f"Invalid value '{short_periods}' for parameter short_periods."
        raise ValueError(msg)

    return smooth_window


def _verify_period_group_sampling(period_group_sampling: int):
    """ Check the period group sampling is valid and reasonable.
    """

    if not (1 <= period_group_sampling <= 9):
        msg = f"Parameter period_group_sampling must be in range [1, 9]."
        raise ValueError(msg)

    return


def _verify_normalisation(normalisation: str):
    """ Check the normalisation is valid.
    """

    if normalisation not in get_args(Normalisation):
        msg = f"Invalid value '{normalisation}' for parameter normalisation."
        raise ValueError(msg)

    return


def _verify_num_processes(num_processes: Optional[int]) -> Optional[int]:
    """ Check that the num_processes parameter is valid and reasonable.
    """

    if num_processes is None:
        return num_processes

    if not isinstance(num_processes, int):
        msg = f"Parameter num_processes must be an integer."
        raise ValueError(msg)

    max_processes = mp.cpu_count()
    if num_processes > mp.cpu_count():
        LOGWARNING(f"Parameter num_processes exceeds mp.cpu_count() = {max_processes}, using that instead.")
        num_processes = max_processes

    return num_processes


def _verify_output_file(output_file: Optional[str]):

    if output_file is None:
        return

    path, filename = os.path.split(output_file)

    _, ext = os.path.splitext(filename)

    if ext != '.asdf':
        msg = f"Output file should be an .asdf file."
        raise ValueError(msg)

    if path != '' and not os.path.exists(path):
        msg = f"Path to {output_file} does not exist."
        raise ValueError(msg)

    return


def load_test_data(lc_name):

    data_file = files('umbra.data').joinpath(lc_name)

    af = asdf.open(data_file)

    return af