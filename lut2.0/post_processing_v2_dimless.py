# post_processing_v2_dimless.py

import numpy as np
import matplotlib.pyplot as plt

from parameters_dimless_v2 import get_dimless_scales

def post_processing_v2_dimless(t_hat, Y_hat):
    """
    Make LUT 2.0 plots in dimensionless form, mirroring Post_Processing_V2.

    t_hat: (n,)   dimensionless time
    Y_hat: (n,10) dimensionless states in the same order as y.
    """

    scales = get_dimless_scales()
    p      = scales["params_dim"]
    Estar  = scales["Estar"]
    V_ref  = scales["V_ref"]
    Q_ref  = scales["Q_ref"]
    N_ref  = scales["N_ref"]
    S      = scales["state_scales"]

    # --- Recover dimensional states for internal formulas ---
    Y_dim = Y_hat * S
    t_dim = t_hat * scales["tau"]      # if you want to compare later

    VB   = Y_dim[:, 0]
    TBA  = Y_dim[:, 1]
    PB   = Y_dim[:, 2]
    NBa  = Y_dim[:, 3]
    lamEUS = Y_dim[:, 4]
    TUA    = Y_dim[:, 5]
    PU     = Y_dim[:, 6]
    NUa    = Y_dim[:, 7]
    # z0B, z0U = Y_dim[:,8], Y_dim[:,9]  # not directly plotted

    # ---- copy of Post_Processing_V2 formulas (dimensional) ----
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

    # Bladder stiffness
    E1  = aExp * np.exp(bExp * VB) + cExp

    # Geometry
    lambdaB = (VB / VB0star)**(1.0/3.0) - 1.0
    RB      = (3.0 * VB / (4.0 * np.pi))**(1.0/3.0)
    TB      = (PB * RB) / (2.0 * hB)

    # Urethra pressures
    p1 = (PB - PU) / 2.0
    p2 = (PU - Pext) / 2.0

    # EUS biomechanics
    Eeus1  = aE * (lamEUS - bE)**0.25 - cE
    Eeus   = Eeus1 + Eeus2

    Aref = np.pi * RUstar**2
    kp   = (Eius / (12.0 * (1.0 - V**2))) * (Aref/np.pi)**(-1.5)
    kL   = 12.0 * (Aref/(np.pi*hU**2))**2

    n = len(PU)
    Rius1 = np.zeros(n)
    Rius2 = np.zeros(n)
    for i in range(n):
        if p1[i] <= pbar:
            Rius1[i] = (8*np.pi*mue1*lu1)/(Aref**2) * (1.0 - p1[i]/kp)**(4.0/3.0)
        else:
            Rius1[i] = (8*np.pi*mue1*lu1)/(Aref**2) * ((p1[i]/(kp*kL))+1.0)**(-4.0)
        if p2[i] <= pbar:
            Rius2[i] = (8*np.pi*mue1*lu3)/(Aref**2) * (1.0 - p2[i]/kp)**(4.0/3.0)
        else:
            Rius2[i] = (8*np.pi*mue1*lu3)/(Aref**2) * ((p2[i]/(kp*kL))+1.0)**(-4.0)

    Reus = (8 * mue1 * lu2) / (np.pi * RUstar**4 * (lamEUS + 1.0)**4)

    f_QB = 1.0 / (1.0 + np.exp(-s * (NBa - NBath_guard)))
    QB   = (PB - PU) / Rius1 * f_QB
    Qout = PU / (Reus + Rius2)

    reus = (lamEUS + 1.0) * RUstar
    TU   = (PU * reus) / hU

    # NBe, NUe_guard
    NBe = np.zeros_like(NBa)
    NUe_guard = np.zeros_like(NUa)

    SB = NBa - NBath_dist
    for j in range(len(NBa)):
        NBe[j] = bn*SB[j] if NBa[j] >= NBath_dist else 0.0

    for j in range(len(NUa)):
        if NUa[j] >= NUath_guard:
            NUe_guard[j] = ba * (NUa[j] - NUath_guard)
        else:
            NUe_guard[j] = 0.0

    # ---------- Now convert everything to dimensionless hats ----------
    VB_hat     = VB / V_ref
    PB_hat     = PB / Estar
    PU_hat     = PU / Estar
    TBA_hat    = TBA / Estar
    TB_hat     = TB / Estar
    TUA_hat    = TUA / Estar
    TU_hat     = TU / Estar
    NBa_hat    = NBa / N_ref
    NBe_hat    = NBe / N_ref
    NUa_hat    = NUa / N_ref
    NUe_hat    = NUe_guard / N_ref
    E1_hat     = E1 / Estar
    Eeus1_hat  = Eeus1 / Estar
    QB_hat     = QB / Q_ref
    Qout_hat   = Qout / Q_ref
    Rius1_hat  = Rius1 / scales["R_ref"]
    Rius2_hat  = Rius2 / scales["R_ref"]
    Reus_hat   = Reus  / scales["R_ref"]
    p1_hat     = p1 / Estar
    p2_hat     = p2 / Estar
    reus_hat   = reus / p["RB0"]      # or /RUstar, 任选一个参考长度
    lambdaB_hat = lambdaB             # already dimensionless
    lamEUS_hat = lamEUS               # already dimensionless

    # ---------- Plot, mirroring your dimensional figure ----------
    fig = plt.figure(figsize=(18, 6))
    # 第一行
    ax = plt.subplot(3, 8, 1);  ax.plot(t_hat, VB_hat);         ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat V_B$ [–]')
    ax = plt.subplot(3, 8, 2);  ax.plot(t_hat, PB_hat);         ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat P_B$ [–]')
    ax = plt.subplot(3, 8, 3);  ax.plot(t_hat, NBa_hat);        ax.axhline(NBath_dist/N_ref, ls='--'); ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat N_{B,a}$ [–]')
    ax = plt.subplot(3, 8, 4);  ax.plot(t_hat, TBA_hat);        ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat T_{B,a}$ [–]')
    ax = plt.subplot(3, 8, 5);  ax.plot(t_hat, lamEUS_hat);     ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\lambda_{\mathrm{EUS}}$ [–]')
    ax = plt.subplot(3, 8, 6);  ax.plot(t_hat, PU_hat);         ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat P_U$ [–]')
    ax = plt.subplot(3, 8, 7);  ax.plot(t_hat, NUa_hat);        ax.axhline(NUath_augm/N_ref, ls='--', c='k'); ax.axhline(NUath_guard/N_ref, ls='--', c='r'); ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat N_{U,a}$ [–]')
    ax = plt.subplot(3, 8, 8);  ax.plot(t_hat, TUA_hat);        ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat T_{U,a}$ [–]')

    # 第二行
    ax = plt.subplot(3, 8, 9);  ax.plot(t_hat, QB_hat);         ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat Q_B$ [–]')
    ax = plt.subplot(3, 8,10);  ax.plot(t_hat, TB_hat);         ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat T_B$ [–]')
    ax = plt.subplot(3, 8,11);  ax.plot(t_hat, NBe_hat);        ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat N_{B,e}$ [–]')
    ax = plt.subplot(3, 8,12);  ax.plot(t_hat, E1_hat);         ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat E_{B1}$ [–]')
    ax = plt.subplot(3, 8,13);  ax.plot(t_hat, Qout_hat);       ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat Q_{\text{out}}$ [–]')
    ax = plt.subplot(3, 8,14);  ax.plot(t_hat, TU_hat);         ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat T_{\mathrm{EUS}}$ [–]')
    ax = plt.subplot(3, 8,15);  ax.plot(t_hat, NUe_hat);        ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat N_{U,e}$ [–]')
    ax = plt.subplot(3, 8,16);  ax.plot(t_hat, Eeus1_hat);      ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat E_{\mathrm{EUS1}}$ [–]')

    # 第三行
    ax = plt.subplot(3, 8,17);  ax.plot(t_hat, f_QB);           ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$f_{QB}$ [–]')
    ax = plt.subplot(3, 8,19);  ax.plot(t_hat, Rius1_hat);      ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat R_{\mathrm{ius1}}$ [–]')
    ax = plt.subplot(3, 8,20);  ax.plot(t_hat, Reus_hat);       ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat R_{\mathrm{eus}}$ [–]')
    ax = plt.subplot(3, 8,21);  ax.plot(t_hat, Rius2_hat);      ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat R_{\mathrm{ius2}}$ [–]')
    ax = plt.subplot(3, 8,22);  ax.plot(t_hat, reus_hat);       ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat r_{\mathrm{EUS}}$ [–]')
    ax = plt.subplot(3, 8,23);  ax.plot(t_hat, p1_hat, label=r'$\hat p_1$'); ax.plot(t_hat, p2_hat, label=r'$\hat p_2$'); ax.axhline(pbar/Estar, label=r'$\hat p_{\mathrm{bar}}$'); ax.set_xlabel(r'$\hat t$'); ax.set_ylabel(r'$\hat p_1, \hat p_2$ [–]')
    # ax.legend(loc="best")

    plt.tight_layout()
    plt.show()
