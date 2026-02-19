from typing import Literal

import numpy as np
from astropy import constants

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

# Literal type hints.
LDType = Literal["uniform", "linear", "quadratic", "square-root", "logarithmic", "exponential", "power2", "nonlinear"]
BinMethod = Literal["points", "window"]
SearchMode = Literal["BLS", "TLS", "WLS"]
ShortPeriods = Literal["skip", "TLS", "WLS"]
SmoothWeights = Literal["uniform", "tricube"]
Normalisation = Literal["simple", "umbra"]