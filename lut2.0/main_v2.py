# main_v2.py

import numpy as np
from scipy.integrate import solve_ivp

from parameters_v2 import get_parameters
from odefcn_v2 import odefcn_v2
from post_processing_v2 import post_processing_v2


def main():
    # Load parameters
    params = get_parameters()

    # Initial conditions (same as Main_V2.m)
    VB0     = 5e-2       # VB0 > 0
    TBA0    = 10e-5
    PB0     = 2000.0     # PB0 > 0
    NBa0    = 0.01       # NBa0 > 0
    lamEUS0 = 0.08
    TUA0    = 10e-5
    PU0     = 2500.0
    NUa0    = 0.01
    z0B0    = 0.1
    z0U0    = 0.1

    y0 = np.array([
        VB0, TBA0, PB0, NBa0, lamEUS0,
        TUA0, PU0, NUa0, z0B0, z0U0
    ], dtype=float)

    # Time span
    Tend = 1000.0  # [s]
    t_span = (0.0, Tend)

    # Tolerances (match MATLAB: RelTol=1e-13, AbsTol vector)
    atol = np.array([
        1e-6, 1e-6, 1e-10, 1e-6, 1e-6,
        1e-6, 1e-6, 1e-6, 1e-6, 1e-6
    ], dtype=float)

    # Optional: specify time points to store
    t_eval = np.linspace(0.0, Tend, 1001)

    # Solve the ODE system using solve_ivp (stiff solver similar to ode15s)
    sol = solve_ivp(
        fun=lambda t, y: odefcn_v2(t, y, params),
        t_span=t_span,
        y0=y0,
        method="BDF",       # stiff solver, analogous to ode15s
        t_eval=t_eval,
        rtol=1e-13,
        atol=atol
    )

    if not sol.success:
        print("Warning: ODE solver did not converge:", sol.message)

    t = sol.t
    y = sol.y.T  # shape (n_times, 10) for easier indexing

    # Post-processing & plotting
    post_processing_v2(t, y, params)


if __name__ == "__main__":
    main()
