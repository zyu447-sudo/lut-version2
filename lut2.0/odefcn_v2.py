# odefcn_v2.py

import numpy as np

def odefcn_v2(t, y, p):
    """
    Python translation of odefcn_V2.m
    Arguments:
      t : float, time
      y : array-like, shape (10,)
      p : dict, parameters from get_parameters()
    Returns:
      dydt : ndarray, shape (10,)
    """
    # Unpack state variables
    VB, TA, PB, NBa, lamEUS, TUA, PU, NUa, z0B, z0U = y

    # --- Unpack parameters (only those used) ---
    Q_in        = p["Q_in"]
    VB0star     = p["VB0star"]
    RB0         = p["RB0"]
    hB          = p["hB"]
    E2          = p["E2"]
    eta         = p["eta"]
    aExp        = p["aExp"]
    bExp        = p["bExp"]
    cExp        = p["cExp"]
    s           = p["s"]

    mue1        = p["mue1"]
    lu1         = p["lu1"]
    lu2         = p["lu2"]
    lu3         = p["lu3"]
    RUstar      = p["RUstar"]
    hU          = p["hU"]

    Eeus2       = p["Eeus2"]
    V           = p["V"]
    aE          = p["aE"]
    bE          = p["bE"]
    cE          = p["cE"]
    Eius        = p["Eius"]

    Cu          = p["Cu"]
    Pext        = p["Pext"]
    pbar        = p["pbar"]

    gamab       = p["gamab"]
    NBath_dist  = p["NBath_dist"]
    bn          = p["bn"]
    NBath_guard = p["NBath_guard"]

    gamau       = p["gamau"]
    NUath_augm  = p["NUath_augm"]
    NUath_guard = p["NUath_guard"]
    ba          = p["ba"]

    kb          = p["kb"]
    vb          = p["vb"]
    m1b         = p["m1b"]
    m2b         = p["m2b"]

    ku          = p["ku"]
    vu          = p["vu"]
    m1u         = p["m1u"]
    m2u         = p["m2u"]

    tetaB       = p["tetaB"]
    aB          = p["aB"]
    rBa         = p["rBa"]
    wB          = p["wB"]

    tetaU       = p["tetaU"]
    aU          = p["aU"]
    rUa         = p["rUa"]
    wU          = p["wU"]

    TauB        = p["TauB"]
    TauU        = p["TauU"]

    # --- Variable bladder geometry ---
    RB = (3.0 * VB / (4.0 * np.pi)) ** (1.0 / 3.0)  # Bladder radius
    lambdaB = (VB / VB0star) ** (1.0 / 3.0) - 1.0   # Bladder stretch

    # --- Urethra pressures ---
    p1 = (PB - PU) / 2.0
    p2 = (PU - Pext) / 2.0

    # --- Variable urethra biomechanics (EUS) ---
    Eeus1   = (aE * (lamEUS - bE) ** 0.25) - cE
    dEeus1  = aE / (4.0 * (lamEUS - bE) ** 0.25)
    Eeus    = Eeus1 + Eeus2   # currently Eeus is not used directly later

    # --- Variable urethra geometry ---
    Aref = np.pi * RUstar**2

    kp = (Eius / (12.0 * (1.0 - V**2))) * (Aref / np.pi) ** (-1.5)
    kL = 12.0 * (Aref / (np.pi * hU**2))**2

    # Resistance in the IUS 1
    if p1 <= pbar:
        Rius1 = (8.0 * np.pi * mue1 * lu1) / (Aref**2) * (1.0 - (p1 / kp)) ** (4.0 / 3.0)
    else:
        Rius1 = (8.0 * np.pi * mue1 * lu1) / (Aref**2) * ((p1 / (kp * kL)) + 1.0) ** (-4.0)

    # Resistance in the IUS 2
    if p2 <= pbar:
        Rius2 = (8.0 * np.pi * mue1 * lu3) / (Aref**2) * (1.0 - (p2 / kp)) ** (4.0 / 3.0)
    else:
        Rius2 = (8.0 * np.pi * mue1 * lu3) / (Aref**2) * ((p2 / (kp * kL)) + 1.0) ** (-4.0)

    # EUS resistance
    Reus = (8.0 * mue1 * lu2) / (np.pi * RUstar**4 * (lamEUS + 1.0)**4)

    # --- Flow ---
    f_QB = 1.0 / (1.0 + np.exp(-s * (NBa - NBath_guard)))  # switch for QB

    QB   = (PB - PU) / Rius1 * f_QB
    Qout = PU / (Reus + Rius2)

    # --- Variable bladder stiffness E1 ---
    E1  = aExp * np.exp(bExp * VB) + cExp
    dE1 = aExp * bExp * np.exp(bExp * VB)

    # --- Efferent bladder signal: distention reflex ---
    if NBa >= NBath_dist:
        NBe_dist = bn * (NBa - NBath_dist)
    else:
        NBe_dist = 0.0

    # --- Efferent urethral augmenting reflex ---
    if NUa >= NUath_augm:
        NUe_aug = ba * (NUa - NUath_augm)
    else:
        NUe_aug = 0.0

    NBe = gamab * NBe_dist + gamau * NUe_aug

    # --- Efferent urethral guarding reflex ---
    if NUa >= NUath_guard:
        NUe_guard = ba * (NUa - NUath_guard)
    else:
        NUe_guard = 0.0

    # --- Coefficients of dPB/dt ---
    a1 = -(E1 / eta)
    a2 = -(E2 / eta)
    a3 = (Q_in - QB) * dE1 / E1
    a4 = -(Q_in - QB) / (3.0 * VB)
    a  = a1 + a2 + a3 + a4

    b1 = ((2.0 * hB * E1) / (3.0 * RB0 * VB)) * (Q_in - QB)
    b2 = ((2.0 * hB * E1 * E2) / (eta * RB)) * ((VB / VB0star) ** (1.0 / 3.0) - 1.0)
    b  = b1 + b2

    c  = (2.0 * hB * E1 * TA) / (eta * RB)

    # --- Coefficients for dlamEUS/dt ---
    # d, e, f as in MATLAB code
    d = (((eta * dEeus1) / (Eeus1**2)) - 1.0 - (Eeus2 / Eeus1)) * (RUstar / hU) * PU
    e = (eta / Eeus1) * (RUstar / hU) * (((PB - PU) / Rius1) - (PU / (Reus + Rius2)))
    f = (eta / Eeus1) * (RUstar / hU) * PU - eta

    # --- ODE system ---
    dydt = np.zeros(10, dtype=float)

    # 1) dVB/dt
    dydt[0] = Q_in - QB

    # 2) dTA/dt (TBA in MATLAB)
    dydt[1] = TA * (lambdaB * NBe - TA) / TauB

    # 3) dPB/dt
    dydt[2] = a * PB + b + c

    # 4) dNBa/dt (afferent bladder neural signal)
    term_pressure = vb * (PB / m1b) ** m2b
    term_dpdt     = wB * max(a * PB + b + c, 0.0) * (1.0 / (z0B / tetaB + 1.0))
    dydt[3] = kb * NBa * (term_pressure + term_dpdt - NBa)

    # 5) dlamEUS/dt
    dydt[4] = (d * (lamEUS + 1.0) - e * (lamEUS + 1.0) + Eeus2 * lamEUS + TUA) / f

    # 6) dTUA/dt
    dydt[5] = (TUA / TauU) * (gamau * lamEUS * NUe_guard - TUA)

    # 7) dPU/dt
    dydt[6] = (((PB - PU) / Rius1) - (PU / (Reus + Rius2))) / Cu

    # 8) dNUa/dt (afferent urethral neural signal)
    term_PU   = vu * (PU / m1u) ** m2u
    term_flow = wU * max((QB - Qout) / Cu, 0.0) * (1.0 / (z0U / tetaU + 1.0))
    dydt[7] = ku * NUa * (term_PU + term_flow - NUa)

    # 9) dz0B/dt
    dydt[8] = aB * (rBa * NBa - z0B)

    # 10) dz0U/dt
    dydt[9] = aU * (rUa * NUa - z0U)

    return dydt
