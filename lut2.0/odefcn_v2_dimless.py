# odefcn_v2_dimless.py
# Dimensionless LUT 2.0: wrapper around dimensional odefcn_v2

import numpy as np
from odefcn_v2 import odefcn_v2
from parameters_dimless_v2 import get_dimless_scales

_SCALES = get_dimless_scales()

def dimless_to_dimensional_state(y_hat, scales=None):
    """Convert dimensionless state y_hat -> dimensional y."""
    if scales is None:
        scales = _SCALES
    S = scales["state_scales"]
    return np.asarray(y_hat, dtype=float) * S

def dimensional_to_dimless_state(y, scales=None):
    """Convert dimensional state y -> dimensionless y_hat."""
    if scales is None:
        scales = _SCALES
    S = scales["state_scales"]
    return np.asarray(y, dtype=float) / S

def odefcn_v2_hat(t_hat, y_hat, scales=None):
    """
    Dimensionless ODE:

      d y_hat / d t_hat = S^{-1} * (dy/dt) * tau

    where t = t_hat * tau, y = S * y_hat.
    """
    if scales is None:
        scales = _SCALES

    tau   = scales["tau"]
    p_dim = scales["params_dim"]
    S     = scales["state_scales"]

    # hat -> dimensional
    t_dim = t_hat * tau
    y_dim = y_hat * S

    # original dimensional RHS
    dydt_dim = odefcn_v2(t_dim, y_dim, p_dim)

    # convert derivative to dimensionless
    dydt_hat = dydt_dim * tau / S
    return dydt_hat
