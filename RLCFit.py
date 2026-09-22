import numpy as np
import matplotlib.pyplot as plt
import Ohm

def make_data_dictionary(t, V, RLC):
    if len(t) != len(V):
        print("Time and Voltage lists need to be of the same length")
        return None

    if len(RLC) != 3 or len(RLC[0]) != 2:
        print("RLC values should be [ [R, dR], L(mH), C(uF) ]")
        return None
    data = dict()

    N = len(V)-1

    t_RLC = np.zeros(N)
    V_RLC = np.zeros(N)
    dt_RLC = np.zeros_like(t_RLC)
    dV_RLC = np.zeros_like(V_RLC)

    for i in range(N):
        t_RLC[i] = t[i][0]
        dt_RLC[i] = t[i][1]
        V_RLC[i] = V[i][0]
        dV_RLC[i] = V[i][1]

    data["t"] = t_RLC
    data["dt"] = dt_RLC
    data["V"] = V_RLC
    data["dV"] = dV_RLC
    
    data["t_unit"] = t[-1]
    data["V_unit"] = V[-1]

    data["t_mult"] = Ohm.unit_conv(t[-1])
    data["V_mult"] = Ohm.unit_conv(V[-1])

    data["R"] = RLC[0][0]
    data["dR"] = RLC[0][1]
    data["L_guess"] = RLC[1]
    data["C_guess"] = RLC[2]

    plt.figure(figsize=(5,3))
    plt.errorbar(t_RLC, V_RLC, yerr=dV_RLC, xerr=dt_RLC, fmt='o',barsabove=True, ms=3, capsize=2, elinewidth=1, color='k')

    plt.xlim(0, np.max(t_RLC)*1.05)

    alpha = data["R"] / (data["L_guess"] * 1e-3) / 2
    omega = 1 / ( (data["L_guess"]*1e-3) * (data["C_guess"] * 1e-6) ) - alpha**2
    omega = np.sqrt(omega)

    c = np.polyfit( data["t"][0::2], np.log(np.abs(data["V"][0::2])), 1)
    V0 = np.exp(c[1])
    alpha = -c[0] / data["t_mult"]

    data["best_guess"] = [V0, alpha, omega]
    tt = np.linspace(0, np.max(t_RLC)*1.05, 1000)
    plt.plot(tt, func_RLC(tt*data["t_mult"], V0, alpha, omega))
    plt.xlabel("time [{}]".format(data["t_unit"]))
    plt.ylabel("V [{}]".format(data["V_unit"]))
    return data

def func_RLC(t, V0, alpha, omega):
    return V0 * np.exp(-alpha*t) * np.sin(omega*t)

def chi_sq_RLC(data, V0, alpha, omega):
    df = func_RLC((data["t"]+data["dt"])*data["t_mult"], V0, alpha, omega)-func_RLC((data["t"]-data["dt"])*data["t_mult"], V0, alpha, omega) 
    df /= 4
    cc = data["V"] - func_RLC(data["t"]*data["t_mult"], V0, alpha, omega)
    cc = cc**2
    cc /= data["dV"]**2 + df**2
    return np.sum(cc)

def chi_V0(data, VV, alpha, omega):
    cc = np.zeros(len(VV))
    for k in range(len(VV)):
        cc[k] = chi_sq_RLC(data, VV[k], alpha, omega)
    ind = np.argmin(cc)
    if ind == 0:
        VVV = np.linspace( VV[0] * VV[0]/VV[-1], VV[1], 10)
        return chi_V0(data, VVV, alpha, omega)
    if ind == len(VV)-1:
        VVV = np.linspace( VV[-2], VV[-1] * VV[-1]/VV[0], 10)
        return chi_V0(data, VVV, alpha, omega)

    if cc[ind-1] < np.min(cc)+1 and cc[ind+1] < np.min(cc)+1:
        return np.min(cc), VV[ind]
    else:
        VVV = np.linspace(VV[ind-1], VV[ind+1], 10)
        return chi_V0(data, VVV, alpha, omega)

def find_best_value(data, c_arr, alph, omg, VV):
    if np.isscalar(alph):
        is_omega = True
        p_arr = np.copy(omg)
    else:
        is_omega = False
        p_arr = np.copy(alph)

    ind = np.where(c_arr < np.min(c_arr)+1)[0]
    if len(ind) < 10:
        if len(ind) == 1:
            pp = np.linspace(p_arr[ind[0]-1], p_arr[ind[0]+1], 50)
        else:
            pp = np.linspace(p_arr[ind[0]], p_arr[ind[-1]], 50)
        cc = np.zeros(len(pp))
        for i in range(len(cc)):
            if is_omega:
                om = pp[i]
                al = alph
            else:
                om = omg
                al = pp[i]
            cc[i], _ = chi_V0(data, VV, al, om)

        if is_omega:
            return find_best_value(data, cc, alph, pp, VV)
        else:
            return find_best_value(data, cc, pp, omg, VV)
    else:
        return p_arr[np.argmin(c_arr)], 0.5*(p_arr[ind[-1]]-p_arr[ind[0]])
        
def best_fit(data, fn=None):
    t = data["t"]
    per = np.array([2*t[1], t[3], t[5]*2/3, t[5]-t[1]])
    om_max = np.max(2*np.pi/(per-np.max(data["dt"])) / data["t_mult"])
    om_min = np.min(2*np.pi/(per+np.max(data["dt"])) / data["t_mult"])

    om = np.linspace(om_min, om_max, 50)
    al = np.linspace(0.8 * data["best_guess"][1], 1.2 * data["best_guess"][1], 50)
    VV = np.linspace(0.9 * data["best_guess"][0], 1.1 * data["best_guess"][0], 10)

    cc = np.zeros((len(om), len(al)))
    V0_mat = np.zeros_like(cc)
    for i in range(len(om)):
        for j in range(len(al)):
            cc[i, j], V0_mat[i,j] = chi_V0(data, VV, al[j], om[i])
                

    ind = np.unravel_index(np.argmin(cc), cc.shape)
    #print(np.min(cc))
    #print(om[ind[0]], al[ind[1]])

    omega_val = om[ind[0]]
    alpha_val = al[ind[1]]
    V0_val = V0_mat[ind[0], ind[1]]

    alpha_val, dalph = find_best_value(data, cc[ind[0],:], al, omega_val, VV)
    omega_val, dom = find_best_value(data, cc[:,ind[1]], alpha_val, om, VV)

    c_val, V0_val = chi_V0(data, VV, alpha_val, omega_val)

    L_val_mH = data["R"] / 2 / alpha_val * 1e3
    dL = (data["dR"] / data["R"])**2 + (dalph/alpha_val)**2
    dL = np.sqrt(dL) * L_val_mH

    w0 = np.sqrt(omega_val**2 + alpha_val**2)
    C_val_uF = 1 / (L_val_mH * 1e-3) / w0**2 * 1e6

    dw0 = (2 * omega_val * dom)**2 + (2 * alpha_val * dalph)**2
    dw0 = np.sqrt(dw0)

    dC = (dL/L_val_mH)**2 + (dw0/w0**2)**2
    dC = np.sqrt(dC)
    dC *= C_val_uF

    print("Best Fit")
    print("L = {} +/- {} mH".format(L_val_mH, dL))
    print("C = {} +/- {} uF".format(C_val_uF, dC))
    print("chi-squared ({} - 2 d.o.f.) = {}".format(len(data["V"]), c_val))

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,5), height_ratios=[2,1], sharex=True)
    plt.subplots_adjust(hspace=0)

    ax[0].errorbar(data["t"], data["V"], yerr=data["dV"], xerr=data["dt"], fmt='o',barsabove=True, ms=3, capsize=2, elinewidth=1, color='k')
    tt = np.linspace(0, np.max(data["t"]), 1000)
    #plt.plot(tt, func_RLC(tt*data["t_mult"], data["best_guess"][0], data["best_guess"][1], om_max))
    #plt.plot(tt, func_RLC(tt*data["t_mult"], data["best_guess"][0], data["best_guess"][1], om_min))
    ax[0].plot(tt, func_RLC(tt*data["t_mult"], V0_val, alpha_val, omega_val))

    residual = data["V"] - func_RLC(data["t"]*data["t_mult"], V0_val, alpha_val, omega_val)

    df = func_RLC((data["t"]+data["dt"])*data["t_mult"], V0_val, alpha_val, omega_val)-func_RLC((data["t"]-data["dt"])*data["t_mult"], V0_val, alpha_val, omega_val) 
    df /= 4
    dr = np.sqrt(data["dV"]**2 + df**2)
    ax[1].errorbar(data["t"], residual, yerr=dr, xerr=data["dt"], fmt='o',barsabove=True, ms=3, capsize=2, elinewidth=1, color='k')
    ax[0].axhline(0, c="0.50", ls="--")
    ax[1].axhline(0, c="0.50", ls="--")

    ax[1].set_xlabel("time [{}]".format(data["t_unit"]), fontsize=12)
    ax[0].set_ylabel("V [{}]".format(data["V_unit"]), fontsize=12)
    ax[1].set_ylabel("residual [{}]".format(data["V_unit"]), fontsize=12)

    if fn is not None:
        plt.savefig(fn+".pdf", bbox_inches='tight')