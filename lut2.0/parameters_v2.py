# parameters_v2.py

import numpy as np

def get_parameters():
    """
    Python translation of Parameters_V2.m
    Returns a dict with all model parameters.
    Units:
      Length: cm
      Volume: mL
      Time: s
      Pressure: Pa
      Voltage: µV
    """
    params = {}

    # Conversion factors
    params["fac_cmH20_Pa"] = 98.0665   # 1 cmH2O = 98.0665 Pa
    params["fac_mmHg_Pa"]  = 133.322   # 1 mmHg  = 133.322 Pa

    # Outside pressures
    params["Pabd"] = 0.0   # [Pa]
    params["Pext"] = 0.0   # [Pa]
    params["pbar"] = 500.0 # [Pa]

    # Inflow rate
    params["Q_in"] = 3.3e-4  # [mL/s]

    # Bladder geometry
    params["VB0star"] = 3e-2  # [mL]
    VB0star = params["VB0star"]
    RB0 = ((3.0 * VB0star) / (4.0 * np.pi)) ** (1.0 / 3.0)
    params["RB0"] = RB0                 # [cm]
    params["LB0"] = 2.0 * np.pi * RB0   # [cm]
    params["hB"]  = RB0 / 20.0          # [cm]

    # Bladder material / viscosity
    params["E2"]  = 20e4        # [Pa]
    params["eta"] = 500e4       # [Pa·s]

    # Variable stiffness of the bladder wall: E1(VB) = aExp * exp(bExp * VB) + cExp
    params["aExp"] = 65.0
    params["bExp"] = 50.0
    params["cExp"] = 8.7e4

    # Sigmoid slopes, etc.
    params["s"]  = 100.0
    params["s2"] = 5.5

    # Urethra fluid viscosity & geometry
    params["mue1"] = 0.000824          # [Pa·s] at 32.2°C
    params["LU"]   = 1.5               # [cm]
    LU = params["LU"]
    params["lu1"]  = LU * (2.0 / 5.0)
    params["lu2"]  = LU * (1.0 / 5.0)
    params["lu3"]  = LU * (2.0 / 5.0)

    params["RUstar"] = 0.007           # [cm]
    params["hU"]     = 0.15            # [cm]

    # Deformable tube parameters / EUS
    params["Eeus2"] = 6.5e7            # [Pa]
    params["V"]     = 0.6              # Poisson ratio

    params["aE"] = 1000.0
    params["bE"] = 0.0
    params["cE"] = 90.0

    params["Eius"] = 1400.0           # [Pa] effective urethral modulus

    # Storage term for urethral volume
    params["Cu"] = 4e-5

    # Bladder neural signal parameters
    params["gamab"]      = 9e5
    params["NBath_dist"] = 0.85
    params["bn"]         = 5.5
    params["NBath_guard"] = 0.90

    params["kb"]  = 1800.0
    params["vb"]  = 0.02            # [µV]
    params["m1b"] = 0.0019 * params["fac_mmHg_Pa"]  # [Pa]
    params["m2b"] = 0.4

    # Urethral neural signal parameters
    params["gamau"]       = 2e5
    params["NUath_augm"]  = 0.65
    params["NUath_guard"] = 0.61
    params["ba"]          = 5.5

    params["ku"]  = 1800.0
    params["vu"]  = 0.02
    params["m1u"] = 0.0019 * params["fac_mmHg_Pa"]
    params["m2u"] = 0.4

    # NBa history
    params["tetaB"] = 3.0
    params["aB"]    = 0.035
    params["rBa"]   = 3.0
    params["wB"]    = 0.0    # NOTE: this disables the dPB/dt term in NBa dynamics

    # NUa history
    params["tetaU"] = 3.0
    params["aU"]    = 0.035
    params["rUa"]   = 3.0
    params["wU"]    = 1e-4

    # TA & TUA ODE time constants
    params["TauB"] = 0.5
    params["TauU"] = 10.0

    return params
