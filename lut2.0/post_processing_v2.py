# post_processing_v2.py

import numpy as np
import matplotlib.pyplot as plt

def post_processing_v2(t, y, p):
    """
    Python translation of Post_Processing_V2.m
    t : array of shape (n_times,)
    y : array of shape (n_times, 10)
    p : parameters dict
    """
    # Unpack parameters used for plotting / derived quantities
    fac_mmHg_Pa = p["fac_mmHg_Pa"]
    VB0star     = p["VB0star"]
    hB          = p["hB"]
    RUstar      = p["RUstar"]
    hU          = p["hU"]
    Eius        = p["Eius"]
    V           = p["V"]
    aE          = p["aE"]
    bE          = p["bE"]
    cE          = p["cE"]
    Eeus2       = p["Eeus2"]
    mue1        = p["mue1"]
    lu1         = p["lu1"]
    lu2         = p["lu2"]
    lu3         = p["lu3"]
    pbar        = p["pbar"]
    Pext        = p["Pext"]
    aExp        = p["aExp"]
    bExp        = p["bExp"]
    cExp        = p["cExp"]
    s           = p["s"]
    NBath_dist  = p["NBath_dist"]
    NBath_guard = p["NBath_guard"]
    NUath_augm  = p["NUath_augm"]
    NUath_guard = p["NUath_guard"]
    bn          = p["bn"]
    ba          = p["ba"]

    # Take state variables (same order as in MATLAB)
    VB  = y[:, 0]
    TBA = y[:, 1]
    PB  = y[:, 2]
    NBa = y[:, 3]
    lamEUS = y[:, 4]
    TUA    = y[:, 5]
    PU     = y[:, 6]
    NUa    = y[:, 7]
    # z0B, z0U not used in plotting

    # Bladder stiffness E1
    E1  = aExp * np.exp(bExp * VB) + cExp
    dE1 = aExp * bExp * np.exp(bExp * VB)   # (not directly plotted, but computed as in MATLAB)

    # Geometry
    lambdaB = (VB / VB0star) ** (1.0 / 3.0) - 1.0
    RB      = (3.0 * VB / (4.0 * np.pi)) ** (1.0 / 3.0)
    TB      = (PB * RB) / (2.0 * hB)

    # Urethra pressures
    p1 = (PB - PU) / 2.0
    p2 = (PU - Pext) / 2.0

    # Variable urethra biomechanics (EUS)
    Eeus1   = (aE * (lamEUS - bE) ** 0.25) - cE
    dEeus1  = aE / (4.0 * (lamEUS - bE) ** 0.25)
    Eeus    = Eeus1 + Eeus2

    # Variable urethra geometry
    Aref = np.pi * RUstar**2

    kp = (Eius / (12.0 * (1.0 - V**2))) * (Aref / np.pi) ** (-1.5)
    kL = 12.0 * (Aref / (np.pi * hU**2))**2

    # Resistances Rius1, Rius2 (time series)
    n = len(PU)
    Rius1 = np.zeros(n)
    Rius2 = np.zeros(n)

    for i in range(n):
        if p1[i] <= pbar:
            Rius1[i] = (8.0 * np.pi * mue1 * lu1) / (Aref**2) * (1.0 - (p1[i] / kp)) ** (4.0 / 3.0)
        else:
            Rius1[i] = (8.0 * np.pi * mue1 * lu1) / (Aref**2) * ((p1[i] / (kp * kL)) + 1.0) ** (-4.0)

    for j in range(n):
        if p2[j] <= pbar:
            Rius2[j] = (8.0 * np.pi * mue1 * lu3) / (Aref**2) * (1.0 - (p2[j] / kp)) ** (4.0 / 3.0)
        else:
            Rius2[j] = (8.0 * np.pi * mue1 * lu3) / (Aref**2) * ((p2[j] / (kp * kL)) + 1.0) ** (-4.0)

    # EUS resistance
    Reus = (8.0 * mue1 * lu2) / (np.pi * RUstar**4 * (lamEUS + 1.0)**4)

    # Flow
    f_QB = 1.0 / (1.0 + np.exp(-s * (NBa - NBath_guard)))
    QB   = (PB - PU) / Rius1 * f_QB
    Qout = PU / (Reus + Rius2)

    # EUS radius and urethral tension
    reus = (lamEUS + 1.0) * RUstar
    TU   = (PU * reus) / hU

    # Distention reflex NBe
    NBe = np.zeros_like(NBa)
    SB  = NBa - NBath_dist
    for j in range(len(NBa)):
        if NBa[j] >= NBath_dist:
            NBe[j] = bn * SB[j]
        else:
            NBe[j] = 0.0

    # Augmenting & guarding reflex
    NUe_aug   = np.zeros_like(NUa)
    NUe_guard = np.zeros_like(NUa)

    for j in range(len(NUa)):
        if NUa[j] >= NUath_augm:
            NUe_aug[j] = ba * (NUa[j] - NUath_augm)
        else:
            NUe_aug[j] = 0.0

        if NUa[j] >= NUath_guard:
            NUe_guard[j] = ba * (NUa[j] - NUath_guard)
        else:
            NUe_guard[j] = 0.0

    # ---------------- Plotting ----------------
    plt.figure(figsize=(16, 8))

    # 1) VB
    plt.subplot(3, 8, 1)
    plt.plot(t, VB)
    plt.xlabel("time [s]")
    plt.ylabel("VB [mL]")

    # 2) PB
    plt.subplot(3, 8, 2)
    plt.plot(t, PB / fac_mmHg_Pa)
    plt.xlabel("time [s]")
    plt.ylabel("PB [mmHg]")

    # 3) NBa
    plt.subplot(3, 8, 3)
    plt.plot(t, NBa)
    plt.axhline(NBath_dist, linestyle="--")
    plt.xlabel("time [s]")
    plt.ylabel("NBa [µV]")

    # 4) TBA
    plt.subplot(3, 8, 4)
    plt.plot(t, TBA)
    plt.xlabel("time [s]")
    plt.ylabel("TBa [Pa]")

    # 5) lamEUS
    plt.subplot(3, 8, 5)
    plt.plot(t, lamEUS)
    plt.xlabel("time [s]")
    plt.ylabel("Stretch [-]")

    # 6) PU
    plt.subplot(3, 8, 6)
    plt.plot(t, PU / fac_mmHg_Pa)
    plt.xlabel("time [s]")
    plt.ylabel("PU [mmHg]")

    # 7) NUa
    plt.subplot(3, 8, 7)
    plt.plot(t, NUa)
    plt.axhline(NUath_augm, linestyle="--", color="k")
    plt.axhline(NUath_guard, linestyle="--", color="r")
    plt.xlabel("time [s]")
    plt.ylabel("NUa [µV]")

    # 8) TUA
    plt.subplot(3, 8, 8)
    plt.plot(t, TUA)
    plt.xlabel("time [s]")
    plt.ylabel("TUA [Pa]")

    # 9) QB
    plt.subplot(3, 8, 9)
    plt.plot(t, QB)
    plt.xlabel("time [s]")
    plt.ylabel("QB [mL/s]")

    # 10) TB
    plt.subplot(3, 8, 10)
    plt.plot(t, TB)
    plt.xlabel("time [s]")
    plt.ylabel("TB [Pa]")

    # 11) NBe
    plt.subplot(3, 8, 11)
    plt.plot(t, NBe)
    plt.xlabel("time [s]")
    plt.ylabel("NBe [µV]")

    # 12) E1
    plt.subplot(3, 8, 12)
    plt.plot(t, E1)
    plt.xlabel("time [s]")
    plt.ylabel("E_B1 [Pa]")

    # 13) Qout
    plt.subplot(3, 8, 13)
    plt.plot(t, Qout)
    plt.xlabel("time [s]")
    plt.ylabel("Qout [mL/s]")

    # 14) TU
    plt.subplot(3, 8, 14)
    plt.plot(t, TU)
    plt.xlabel("time [s]")
    plt.ylabel("Teus [Pa]")

    # 15) NUe_guard
    plt.subplot(3, 8, 15)
    plt.plot(t, NUe_guard)
    plt.xlabel("time [s]")
    plt.ylabel("NUe [µV]")

    # 16) Eeus1
    plt.subplot(3, 8, 16)
    plt.plot(t, Eeus1)
    plt.xlabel("time [s]")
    plt.ylabel("Eeus1 [Pa]")

    # 17) f_QB
    plt.subplot(3, 8, 17)
    plt.plot(t, f_QB)
    plt.xlabel("time [s]")
    plt.ylabel("fQB")

    # 19) Rius1
    plt.subplot(3, 8, 19)
    plt.plot(t, Rius1)
    plt.xlabel("time [s]")
    plt.ylabel("Rius1 [-]")

    # 20) Reus
    plt.subplot(3, 8, 20)
    plt.plot(t, Reus)
    plt.xlabel("time [s]")
    plt.ylabel("Reus [-]")

    # 21) Rius2
    plt.subplot(3, 8, 21)
    plt.plot(t, Rius2)
    plt.xlabel("time [s]")
    plt.ylabel("Rius2 [-]")

    # 22) reus (EUS radius)
    plt.subplot(3, 8, 22)
    plt.plot(t, reus)
    plt.xlabel("time [s]")
    plt.ylabel("EUS Radius [cm]")

    # 23) p1 & p2
    plt.subplot(3, 8, 23)
    plt.plot(t, p1 / fac_mmHg_Pa, label="p1")
    plt.plot(t, p2 / fac_mmHg_Pa, label="p2")
    plt.axhline(pbar / fac_mmHg_Pa, label="pbar")
    plt.xlabel("time [s]")
    plt.ylabel("P1 & P2 [mmHg]")
    # plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
