# parameters_dimless_v2.py
# Build dimensionless scales and Pi-groups for LUT 2.0

import numpy as np
from parameters_v2 import get_parameters

def get_dimless_scales():
    """
    Construct basic scales + state scaling for LUT 2.0.

    Scales:
      Time:   t_hat = t / tau,   tau = eta / E2
      Volume: VB_hat = VB / V_ref,    V_ref = VB0star
      Stress: *_hat = */E2
      Neural: N_hat = N / N_ref,      N_ref = NBath_guard
    """
    p = get_parameters()

    # Basic physical scales
    Estar = p["E2"]                   # Pa
    tau   = p["eta"] / Estar          # s
    V_ref = p["VB0star"]              # mL
    N_ref = p["NBath_guard"]          # uV

    # Derived scales
    Q_ref = V_ref / tau               # mL/s  (volume / time)
    R_ref = Estar / Q_ref             # Pa·s/mL  (resistance scale)

    # State scaling S (same order as y in odefcn_v2)
    state_scales = np.array([
        V_ref,   # VB
        Estar,   # TBA
        Estar,   # PB
        N_ref,   # NBa
        1.0,     # lamEUS (already dimensionless)
        Estar,   # TUA
        Estar,   # PU
        N_ref,   # NUa
        N_ref,   # z0B
        N_ref,   # z0U
    ], dtype=float)

    # Some useful Pi-groups for later (optional but handy)
    Pi_Qin   = p["Q_in"] * tau / V_ref          # inflow
    Pi_hB    = p["hB"] / p["RB0"]               # wall thickness vs radius
    Pi_gammaB = p["gamab"] * N_ref / Estar
    Pi_gammaU = p["gamau"] * N_ref / Estar

    scales = {
        "params_dim": p,
        "Estar": Estar,
        "tau": tau,
        "V_ref": V_ref,
        "N_ref": N_ref,
        "Q_ref": Q_ref,
        "R_ref": R_ref,
        "state_scales": state_scales,
        "Pi_Qin": Pi_Qin,
        "Pi_hB": Pi_hB,
        "Pi_gammaB": Pi_gammaB,
        "Pi_gammaU": Pi_gammaU,
    }
    return scales
