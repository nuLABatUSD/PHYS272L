import numpy as np
import matplotlib.pyplot as plt

def unit_conv(unit):
    if unit[0] == 'm' and (len(unit) > 1):
        return 1e-3
    elif unit[0] == 'u':
        return 1e-6
    elif unit[0] == 'c':
        return 1e-2
    else:
        return 1

def N_slit_maxima_m(m, lam_nm, L_m, d_mm):
    th = np.arcsin(m * lam_nm * 1e-9 / (d_mm * 1e-3))
    return L_m * np.tan(th)

def make_data_dictionary(slit_data, L_data, d_data, lam_guess):
    N = len(slit_data)-1

    m = np.zeros(N, dtype = int)
    y = np.zeros(N)
    dy = np.zeros_like(y)

    for i in range(N):
        m[i] = slit_data[i][0]
        y[i] = slit_data[i][1]
        dy[i] = slit_data[i][2]


    data = dict()
    data["m"] = m
    data["y"] = y
    data["dy"] = dy

    data["y_unit"] = slit_data[-1]
    data["y_unit_mult"] = unit_conv(data["y_unit"])

    data["L"] = L_data[0]
    data["dL"] = L_data[1]
    data["L_unit"] = L_data[2]
    data["L_unit_mult"] = unit_conv(data["L_unit"])

    data["d"] = d_data[0]
    data["dd"] = d_data[1]
    data["d_unit"] = d_data[2]
    data["d_unit_mult"] = unit_conv(data["d_unit"])

    data["lambda_guess_nm"] = lam_guess

    plt.figure(figsize=(4,3))
    mm = np.linspace(0., np.max(data["m"]), 100)
    plt.plot(mm, N_slit_maxima_m(mm, data["lambda_guess_nm"], data["L"] * data["L_unit_mult"], data["d"] * data["d_unit_mult"] * 1000) / data["y_unit_mult"])
    plt.errorbar(data["m"], data["y"], yerr=data["dy"], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    plt.ylabel("y [{}]".format(data["y_unit"]), fontsize=13)
    plt.xlabel("m", fontsize=13)
    plt.ylim(-1, 1.1*np.max(data["y"]))
    plt.xlim(-1, np.max(data["m"])+1)

    return data

def chi_sq_N_slit(data, y0_mm, lam_nm):
    cc = (data["y"] - y0_mm / 1000 / data["y_unit_mult"])- data["y_unit_mult"]**-1 * N_slit_maxima_m(data["m"], lam_nm, data["L"] * data["L_unit_mult"], data["d"] * data["d_unit_mult"] * 1e3)
    cc = cc**2

    th = np.arcsin(data["m"] * lam_nm * 1e-9 / (data["d"] * data["d_unit_mult"]))

    delta = data["dy"]**2
    delta += (np.tan(th) * data["dL"] * data["L_unit_mult"] / data["y_unit_mult"])**2
    delta += (data["y"] / np.cos(th)**2 * (data["dd"] * data["d"]))**2

    cc = cc / delta
    return np.sum(cc)

def N_slit_best_fit(data, fn=None):
    dy_max = np.max(data["dy"]) * data["y_unit_mult"] * 1e3
    y_mult = 2
    l_mult = 0.5
    lmid = data["lambda_guess_nm"]
    for k in range(5):
        y0_mm = np.linspace(-y_mult*dy_max, y_mult*dy_max, 50)
        l_nm = np.linspace((1-l_mult)*lmid, (1+l_mult)*lmid, 100)
    
        cc = np.zeros((len(l_nm), len(y0_mm)))
        for i in range(len(l_nm)):
            for j in range(len(y0_mm)):
                cc[i, j] = chi_sq_N_slit(data, y0_mm[j], l_nm[i])
    
        #plt.plot(y0_mm, np.min(cc, axis=0))
        #plt.plot(l_nm, np.min(cc, axis=1))
        cc_y = np.min(cc, axis=0)
        cc_l = np.min(cc, axis=1)
        
        y0_best = y0_mm[np.argmin(cc_y)]
        l_best = l_nm[np.argmin(cc_l)]
    
        y_ind = np.where(cc_y < np.min(cc_y)+1)[0]
        l_ind = np.where(cc_l < np.min(cc_l)+1)[0]

        if(len(y_ind) < 5):
            y_mult *= 0.5
        if(len(l_ind) < 5):
            l_mult *= 0.5

        lmid = l_best

        if len(y_ind) >= 5 and len(l_ind) >= 5:
            break

    print("Best fit:")
    #print("y_0 = {} +/- {} mm".format(y0_best, 0.5*(y0_mm[y_ind[-1]]-y0_mm[y_ind[0]])))
    print("lambda = {} +/- {} nm".format(l_best, 0.5*(l_nm[l_ind[-1]]-l_nm[l_ind[0]])))
    print("=========")
    print("chi-squared ({}-2 d.o.f) = {}".format(len(data["y"]), np.min(cc)))

    plt.figure(figsize=(4,3))
    mm = np.linspace(0., np.max(data["m"]), 100)
    plt.plot(mm, y0_best / 1000 / data["y_unit_mult"]+ N_slit_maxima_m(mm, l_best, data["L"] * data["L_unit_mult"], data["d"] * data["d_unit_mult"] * 1000) / data["y_unit_mult"])
    plt.errorbar(data["m"], data["y"], yerr=data["dy"], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    plt.ylabel("y [{}]".format(data["y_unit"]), fontsize=13)
    plt.xlabel("m", fontsize=13)
    plt.ylim(-1, 1.1*np.max(data["y"]))
    plt.xlim(-1, np.max(data["m"])+1)

    y0 = [y0_best, 0.5*(y0_mm[y_ind[-1]]-y0_mm[y_ind[0]])]
    l = [l_best, 0.5*(l_nm[l_ind[-1]]-l_nm[l_ind[0]])]

    return y0, l

def make_ruler_dictionary(y_ruler, y_level, L, d_inches, lambda_nm):
    N = len(y_ruler)-1

    m_r = np.zeros(N, dtype=int)
    y_r = np.zeros(N)
    dy_r = np.zeros(N)

    for i in range(N):
        m_r[i] = y_ruler[i][0]
        y_r[i] = y_ruler[i][1]
        dy_r[i] = y_ruler[i][2]

    data = dict()

    data["m_r"] = m_r
    data["y_r"] = y_r
    data["dy_r"] = dy_r
    
    data["y_r_unit"] = y_ruler[-1]
    data["y_r_unit_mult"] = unit_conv(y_ruler[-1])

    data["y_level"] = y_level[0]
    data["dy_level"] = y_level[1]
    data["y_level_unit"] = y_level[2]
    data["y_level_unit_mult"] = unit_conv(y_level[2])

    data["L_unit"] = L[2]
    data["L_unit_mult"] = unit_conv(L[2])
    data["L_m"] = L[0] * data["L_unit_mult"]
    data["dL_m"] = L[1] * data["L_unit_mult"]

    data["d_mm"] = d_inches * 25.4
    data["lambda_guess"] = lambda_nm

    try:
        y0 = y_r[np.where(m_r==0)][0]
    except:
        print("you need to include a m=0 data point. Abort.")
        return None
    
    data["alpha_guess"] = 0.5 * np.arctan((y0 * data["y_r_unit_mult"] - y_level[0] * data["y_level_unit_mult"])/(L[0] * data["L_unit_mult"]))

    mm = np.linspace(np.min(data["m_r"]), np.max(data["m_r"]),100)

    plt.figure(figsize=(4,3))
    plt.plot(mm, ruler_val_m(mm, data["d_mm"], data["L_m"], data["y_level"] * data["y_level_unit_mult"], data["alpha_guess"], data["lambda_guess"]) / data["y_r_unit_mult"])
    plt.errorbar(data["m_r"], data["y_r"], yerr=data["dy_r"], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    plt.ylabel("y [{}]".format(data["y_r_unit"]), fontsize=13)
    plt.xlabel("m", fontsize=13)

    return data

def ruler_val_m(m, d_mm, L_m, y_l_m, alph, lam_nm):
    beta = np.cos(alph) - m * lam_nm / d_mm * 1e-6
    beta = np.arccos(beta)
    return y_l_m + L_m * np.tan(alph + beta)

def make_initial_ruler_plot(data):
    mm = np.linspace(np.min(data["m_r"]), np.max(data["m_r"]),100)

    plt.plot(mm, ruler_val_m(mm, data["d_mm"], data["L_m"], data["y_level"] * data["y_level_unit_mult"], data["alpha_guess"], data["lambda_guess"]) / data["y_r_unit_mult"])
    plt.errorbar(data["m_r"], data["y_r"], yerr=data["dy_r"], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')

def chi_sq_ruler(data, alph, lam):
    cc = (data["y_r"] * data["y_r_unit_mult"] - ruler_val_m(data["m_r"], data["d_mm"], data["L_m"], data["y_level"] * data["y_level_unit_mult"], alph, lam))**2

    beta = np.cos(alph) - data["m_r"] * lam / data["d_mm"] * 1e-6
    beta = np.arccos(beta)

    delta = (data["dy_r"] * data["y_r_unit_mult"])**2
    delta += (data["dy_level"] * data["y_level_unit_mult"])**2
    delta += (data["dL_m"]  * np.tan(alph + beta))**2

    return np.sum(cc/delta)

def ruler_best_fit(data, fn=None):
    alph_mult = 0.25
    l_mult = 0.5
    lmid = data["lambda_guess"]
    amid = data["alpha_guess"]

    a_best = np.zeros(100)
    for k in range(5):
        l_nm = np.linspace((1-l_mult)*lmid, (1+l_mult)*lmid, 100)
        cc_l = np.zeros(len(l_nm))
    
        for i in range(len(l_nm)):
            if np.min(data["m_r"]) < 0:
                a_min = np.arccos(1 + np.min(data["m_r"]) * l_nm[i] / data["d_mm"] * 1e-6)
            else:
                a_min = 0
            a_rad = np.linspace(max(a_min, (1-alph_mult)*amid), (1+alph_mult)*amid, 100)
            cc_a = np.zeros(len(a_rad))
            for j in range(len(a_rad)):
                cc_a[j] = chi_sq_ruler(data, a_rad[j], l_nm[i])

            cc_l[i] = np.min(cc_a)
            a_best[i] = a_rad[np.argmin(cc_a)]

        l_best = l_nm[np.argmin(cc_l)]
        l_ind = np.where(cc_l < np.min(cc_l)+1)[0]

        a_fit = a_best[np.argmin(cc_l)]

        if(len(l_ind) < 5):
            l_mult *= 0.5

        lmid = l_best

        if len(l_ind) >= 5:
            break

    amid = data["alpha_guess"]
        
    alph_mult=0.5

    for k in range(5):
        if np.min(data["m_r"]) < 0:
            a_min = np.arccos(1 + np.min(data["m_r"]) * l_best / data["d_mm"] * 1e-6)
        else:
            a_min = 0
        a_rad = np.linspace(max(a_min, (1-alph_mult)*amid), (1+alph_mult)*amid, 100)   
        cc = np.zeros_like(a_rad)
        for i in range(len(a_rad)):
            cc[i] = chi_sq_ruler(data, a_rad[i], l_best)
    
        a_best = a_rad[np.argmin(cc)]
        a_ind = np.where(cc < np.min(cc)+1)[0]
        
        if len(a_ind) < 5:
            alph_mult *= 0.5
        else:
            break

    print("Best fit:")
    print("lambda = {} +/- {} nm".format(l_best, 0.5*(l_nm[l_ind[-1]]-l_nm[l_ind[0]])))
    print("alpha = {} +/- {} degrees".format(a_best*180/np.pi, 0.5*(a_rad[a_ind[-1]]-a_rad[a_ind[0]])*180/np.pi))
#    print("=========")
    print("chi-squared ({}-2 d.o.f) = {}".format(len(data["y_r"]), np.min(cc_l)))

    mm = np.linspace(np.min(data["m_r"]), np.max(data["m_r"]),100)

    plt.figure(figsize=(4,3))
    plt.plot(mm, ruler_val_m(mm, data["d_mm"], data["L_m"], data["y_level"] * data["y_level_unit_mult"], a_fit, l_best) / data["y_r_unit_mult"])
    plt.errorbar(data["m_r"], data["y_r"], yerr=data["dy_r"], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    plt.ylabel("y [{}]".format(data["y_r_unit"]), fontsize=13)
    plt.xlabel("m", fontsize=13)

    a = [a_best, 0.5*(a_rad[a_ind[-1]]-a_rad[a_ind[0]])]
    l = [l_best, 0.5*(l_nm[l_ind[-1]]-l_nm[l_ind[0]])]
    return a, l


def global_fit(Ndata, Rdata, res_s, res_r, fn=None):    
    N_total = len(Ndata["m"]) + len(Rdata["m_r"])

    y0_arr = np.linspace(res_s[0][0]- 2*res_s[0][1], res_s[0][0]+ 2*res_s[0][1], 25)
    alph_arr = np.linspace(res_r[0][0] - 2*res_r[0][1],res_r[0][0] + 2*res_r[0][1], 25)

    lam_arr = np.linspace(min(res_s[1][0] - 2*res_s[1][1], res_r[1][0] - 2*res_r[1][1]), max(res_s[1][0] + 2*res_s[1][1], res_r[1][0] + 2*res_r[1][1]), 50)


    for ii in range(5):
        cc = np.zeros_like(lam_arr)
        for i in range(len(cc)):
            cc_ya = np.zeros((len(y0_arr), len(alph_arr)))
            for j in range(len(y0_arr)):
                for k in range(len(alph_arr)):
                    cc_ya[j, k] = chi_sq_N_slit(Ndata, y0_arr[j], lam_arr[i])
                    cc_ya[j, k] += chi_sq_ruler(Rdata, alph_arr[k], lam_arr[i])
            cc[i] = np.min(cc_ya)

        
        if ii == 0:
            lam_ind = np.where(cc < max(5 * N_total, 2 * np.min(cc)))[0]
            lam_arr = np.linspace(lam_arr[lam_ind[0]], lam_arr[lam_ind[-1]], 100)
        else:
            l_best = lam_arr[np.argmin(cc)]
            l_ind = np.where(cc < np.min(cc)+1)[0]
            dl_best = 0.5*(lam_arr[l_ind[-1]]-lam_arr[l_ind[0]])
            if len(l_ind) < 5:
                delta_l = 0.5*(lam_arr[-1] - lam_arr[0])
                delta_l *= 0.5
                lam_arr = np.linspace(l_best-delta_l, l_best+delta_l, 100)
            else:
                break    

    print("Best Fit")
    print("lambda = {} +/- {} nm".format(l_best, dl_best))
    print("chi-squared ({}-3 d.o.f) = {}".format(N_total, np.min(cc)))
    
    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)

    ax[0].errorbar(Ndata["m"], Ndata["y"], yerr=Ndata["dy"], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    ax[1].errorbar(Rdata["m_r"], Rdata["y_r"], yerr=Rdata["dy_r"], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')

    ax[0].set_ylabel("y [{}]".format(Ndata["y_unit"]), fontsize=12)
    ax[0].set_xlabel("m", fontsize=12)
    ax[1].set_ylabel("y [{}]".format(Rdata["y_r_unit"]), fontsize=12)
    ax[1].set_xlabel("m", fontsize=12)
    
    mm = np.linspace(np.min(Ndata["m"]), np.max(Ndata["m"]), 100)
    ax[0].plot(mm, res_s[0][0] / 1000 / Ndata["y_unit_mult"]+ N_slit_maxima_m(mm, l_best, Ndata["L"] * Ndata["L_unit_mult"], Ndata["d"] * Ndata["d_unit_mult"] * 1000) / Ndata["y_unit_mult"])
    ax[0].set_title("N-slit interference")
    
    mmr = np.linspace(np.min(Rdata["m_r"]), np.max(Rdata["m_r"]), 100)
    ax[1].plot(mmr, ruler_val_m(mmr, Rdata["d_mm"], Rdata["L_m"], Rdata["y_level"] * Rdata["y_level_unit_mult"], res_r[0][0], l_best) / Rdata["y_r_unit_mult"])
    ax[1].set_title("Reflection interference")
    if fn is not None:
        plt.savefig(fn+".pdf", bbox_inches='tight')