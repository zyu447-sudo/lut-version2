# main_dimless_v2.py

import numpy as np
from scipy.integrate import solve_ivp

from parameters_dimless_v2 import get_dimless_scales
from odefcn_v2_dimless import odefcn_v2_hat, dimensional_to_dimless_state
from post_processing_v2_dimless import post_processing_v2_dimless

def main():
    scales = get_dimless_scales()
    tau    = scales["tau"]

    # ---- dimensional initial conditions (same as Main_V2.m) ----
    VB0     = 5e-2
    TBA0    = 10e-5
    PB0     = 2000.0
    NBa0    = 0.01
    lamEUS0 = 0.08
    TUA0    = 10e-5
    PU0     = 2500.0
    NUa0    = 0.01
    z0B0    = 0.1
    z0U0    = 0.1

    y0_dim = np.array([VB0, TBA0, PB0, NBa0, lamEUS0,
                       TUA0, PU0, NUa0, z0B0, z0U0], dtype=float)

    # ---- convert IC to dimensionless ----
    y0_hat = dimensional_to_dimless_state(y0_dim, scales)

    # ---- choose physical horizon & convert to hat-time ----
    Tend_dim    = 1000.0       # seconds (same as your dimensional run)
    t_hat_final = Tend_dim / tau

    t_hat_eval = np.linspace(0.0, t_hat_final, 1001)

    sol = solve_ivp(
        fun=lambda th, yh: odefcn_v2_hat(th, yh, scales),
        t_span=(0.0, t_hat_final),
        y0=y0_hat,
        t_eval=t_hat_eval,
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        print("WARNING: solver failed:", sol.message)

    t_hat = sol.t
    Y_hat = sol.y.T

    # ---- make the dimensionless version of your big figure ----
    post_processing_v2_dimless(t_hat, Y_hat)

if __name__ == "__main__":
    main()
