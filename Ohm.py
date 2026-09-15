import numpy as np
import matplotlib.pyplot as plt

def unit_conv(unit):
    if unit[0] == 'm':
        return 1e-3
    elif unit[0] == 'u':
        return 1e-6
    else:
        return 1

def make_data_dictionary(voltage, current, guess):
    if len(voltage) != len(current):
        print("Voltage and Current lists need to be of the same length")
        return None
        
    data = dict()

    N_ohm = len(voltage)-1

    V_R = np.zeros(N_ohm)
    dV_R = np.zeros_like(V_R)
    I_R = np.zeros_like(V_R)
    dI_R = np.zeros_like(V_R)

    for i in range(N_ohm):
        V_R[i] = voltage[i][0]
        dV_R[i] = voltage[i][1]
        I_R[i] = current[i][0]
        dI_R[i] = current[i][1]
    
    data["V_R"] = V_R
    data["I_R"] = I_R
    data["dV_R"] = dV_R
    data["dI_R"] = dI_R


    data["V_R_unit"] = voltage[-1]
    data["I_R_unit"] = current[-1]

    data["V_R_mult"] = unit_conv(voltage[-1])
    data["I_R_mult"] = unit_conv(current[-1])

    data["R_guess"] = guess

    plt.figure(figsize=(5,3))
    plt.errorbar(V_R, I_R, xerr=dV_R, yerr=dI_R, fmt='o',barsabove=True, ms=3, capsize=2, elinewidth=1, color='k')

    plt.xlim(0, np.max(V_R)*1.05)
    
    xx = np.linspace(0, np.max(V_R)*1.05)
    plt.plot(xx, xx/guess/data["I_R_mult"]*data["V_R_mult"])
    plt.xlabel("Voltage [{}]".format(data["V_R_unit"]), fontsize=12)
    plt.ylabel("Current [{}]".format(data["I_R_unit"]), fontsize=12)

    plt.ylim(0, np.max(I_R+dI_R)*1.05)
    return data


def chi_square_R(data, R):
    chi = (data["I_R"] * data["I_R_mult"] - data["V_R"] * data["V_R_mult"] / R)
    chi = chi**2  / (data["dI_R"]**2 * data["I_R_mult"]**2 + data["dV_R"]**2 * data["V_R_mult"]**2 / R**2)
    return np.sum(chi)

def set_chi_R(data, RR):
    cc = np.zeros_like(RR)
    for i in range(len(RR)):
        cc[i] = chi_square_R(data, RR[i])
    return cc
    
def R_best_fit(data, filename=None):
    RR = np.linspace(data["R_guess"]*0.9, data["R_guess"]*1.1, 100)
    cc = set_chi_R(data, RR)

    while np.argmin(cc) == 0 or np.argmin(cc) == len(cc) - 1:
        if np.argmin(cc) == 0:
            RR = np.linspace(RR[0] * 0.5, RR[0], 100)
        else:
            RR = np.linspace(RR[-1], RR[-1] * 2, 100)
        cc = set_chi_R(data, RR)

    ind = np.where(cc < np.min(cc) + 10)[0]
    while len(ind) < 40:
        RR = np.linspace(RR[ind[0]], RR[ind[-1]], 50)
        cc = set_chi_R(data, RR)
        ind = np.where(cc < np.min(cc) + 10)[0]

    R_best = RR[np.argmin(cc)]
    ind = np.where(cc < np.min(cc)+1)[0]
    dR_best = 0.5*(RR[ind[-1]]-RR[ind[0]])
    print("Best Fit")
    print("R = {} +/- {} Ohms".format(R_best, dR_best))
    print("chi-squared ({} - 1 d.o.f.) = {}".format(len(data["V_R"]), np.min(cc)))
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,5), height_ratios=[2,1], sharex=True)
    plt.subplots_adjust(hspace=0)

    ax[0].errorbar(data["V_R"], data["I_R"], xerr=data["dV_R"], yerr=data["dI_R"], fmt='o',barsabove=True, ms=3, capsize=2, elinewidth=1, color='k')

    ax[0].set_xlim(0, np.max(data["V_R"])*1.05)
    
    xx = np.linspace(0, np.max(data["V_R"])*1.05)
    ax[0].plot(xx, xx/R_best/data["I_R_mult"]*data["V_R_mult"])
    ax[1].set_xlabel("Voltage [{}]".format(data["V_R_unit"]), fontsize=12)
    ax[0].set_ylabel("Current [{}]".format(data["I_R_unit"]), fontsize=12)

    ax[0].set_ylim(0, np.max(data["I_R"]+data["dI_R"])*1.05)

    ax[1].axhline(0, ls="--", c="0.50")
    residual = data["I_R"] - data["V_R"] * data["V_R_mult"] / data["I_R_mult"] / R_best
    dres = data["dI_R"]**2 + (data["dV_R"] * data["V_R_mult"] / data["I_R_mult"] / R_best)**2
    dres = np.sqrt(dres)
    ax[1].errorbar(data["V_R"], residual, xerr=data["dV_R"], yerr = dres, fmt='o',barsabove=True, ms=3, capsize=2, elinewidth=1, color='k')

    ax[1].set_ylabel("Residual [{}]".format(data["I_R_unit"]), fontsize=12)

    if filename != None:
        plt.savefig(filename+".pdf", bbox_inches='tight')