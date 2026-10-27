import numpy as np
import matplotlib.pyplot as plt

def make_data_dictionary(gas_list):
    N_elements = len(gas_list)
    N_data = 0
    for res in gas_list:
        N_data += len(res) - 1

    NIST_nm = np.zeros(N_data)
    measured_nm = np.zeros_like(NIST_nm)
    dlam_nm = np.zeros_like(NIST_nm)

    i = 0

    elements = []
    el_list = np.zeros(N_data, dtype=int)
    for ii, res in enumerate(gas_list):
        for j in range(len(res)-1):
            NIST_nm[i] = res[j][0]
            measured_nm[i] = res[j][1]
            dlam_nm[i] = res[j][2]
            el_list[i] = ii
            i += 1
        elements.append(res[-1])

    data = dict()
    data["NIST_nm"] = NIST_nm
    data["measured_nm"] = measured_nm
    data["dlam_nm"] = dlam_nm
    data["element_index"] = el_list
    data["elements_list"] = elements

    plt.figure()

    marker = ["o", "x", "*", "^"]
    for i in range(len(marker)):
        for j in range(len(data["measured_nm"])):
            if data["element_index"][j] == i:
                plt.plot(data["measured_nm"][j], (data["NIST_nm"] - data["measured_nm"])[j], marker[data["element_index"][j]], ms=5, color='k', label=data["elements_list"][i])
                break

    for i in range(len(data["measured_nm"])):
        plt.errorbar(data["measured_nm"][i], (data["NIST_nm"] - data["measured_nm"])[i], yerr=data["dlam_nm"][i], xerr=data["dlam_nm"][i], fmt=marker[data["element_index"][i]], ms=5, barsabove=True, capsize=2, elinewidth=1, color='k')
    plt.ylabel(r"$\lambda_{\rm true} - \lambda_{\rm meas}$ [nm]")
    plt.xlabel(r"$\lambda_{\rm meas}$ [nm]")
    plt.legend()
    
    return(data)


def chi_sq(data, A, B):
    cc = (data["NIST_nm"] - data["measured_nm"]) - (A * (data["measured_nm"]) + B)
    cc = cc**2

    delta = data["dlam_nm"]**2
    delta += A**2 * data["dlam_nm"]**2

    return np.sum(cc/delta)

def chi_sq_matrix(data, AA, BB):
    cc = np.zeros((len(AA), len(BB)))
    
    for i in range(len(AA)):
        for j in range(len(BB)):
            cc[i,j] = chi_sq(data, AA[i], BB[j])
    return cc
    
def best_fit(data, fn=None):
    delta_lam = np.max(data["measured_nm"]) - np.min(data["measured_nm"])
    BB = np.linspace(-5, 5, 25)
    AA = np.linspace(-10/delta_lam, 10/delta_lam, 25)
    dB = 0.5 * (BB[-1] - BB[0])
    dA = 0.5 * (AA[-1] - AA[0])

    for k in range(10):
        cc = chi_sq_matrix(data, AA, BB)
    
        index = np.unravel_index(np.argmin(cc), cc.shape)
        
        if(len(np.where(np.min(cc, axis=0) < np.min(cc)+1)[0]) < 10):
            dB_new = 0.5 * dB
        else:
            dB_new = dB
        if len(np.where(np.min(cc, axis=1) < np.min(cc)+1)[0]) < 10:
            dA_new = 0.5 * dA
        else:
            dA_new = dA

        if dB_new == dB and dA_new == dA:
            break

        dB = dB_new
        dA = dA_new
        
        BB = np.linspace(BB[index[1]] - dB, BB[index[1]] + dB, 100)
        AA = np.linspace(AA[index[0]] - dA, AA[index[0]] + dA, 100)

    B_index = np.where(np.min(cc, axis=0) < np.min(cc) + 7)[0]
    BB = np.linspace(BB[B_index[0]], BB[B_index[-1]], 100)
    A_index = np.where(np.min(cc, axis=1) < np.min(cc) + 7)[0]
    AA = np.linspace(AA[A_index[0]], AA[A_index[-1]], 100)

    cc = chi_sq_matrix(data, AA, BB)
    c_index = np.unravel_index(np.argmin(cc), cc.shape)

    A_index = np.where(np.min(cc, axis=1) < np.min(cc) + 1)[0]
    B_index = np.where(np.min(cc, axis=0) < np.min(cc) + 1)[0]

    deltaA = 0.5*(AA[A_index[-1]] - AA[A_index[0]])
    deltaB = 0.5*(BB[B_index[-1]] - BB[B_index[0]])

    plt.figure(figsize=(5,3))

    marker = ["o", "x", "*", "^"]

    for i in range(len(marker)):
        for j in range(len(data["measured_nm"])):
            if data["element_index"][j] == i:
                plt.plot(data["measured_nm"][j], (data["NIST_nm"] - data["measured_nm"])[j], marker[data["element_index"][j]], ms=5, color='k', label=data["elements_list"][i])
                break
    ll = np.linspace(min(400, np.min(data["measured_nm"])*0.95), max(700, np.max(data["measured_nm"])*1.05), 10)
    for i in range(len(data["measured_nm"])):
        plt.errorbar(data["measured_nm"][i], (data["NIST_nm"] - data["measured_nm"])[i], yerr=data["dlam_nm"][i], xerr=data["dlam_nm"][i], fmt=marker[data["element_index"][i]], ms=5, barsabove=True, capsize=2, elinewidth=1, color='k')
    plt.ylabel(r"$\Delta \lambda$ [nm]", fontsize=13)
    plt.xlabel(r"$\lambda_{\rm meas}$ [nm]", fontsize=13)
    plt.plot(ll, AA[c_index[0]] * ll + BB[c_index[1]])
    plt.legend()
    print("Best Fit")
    print("A = {} +/- {}".format(AA[c_index[0]], deltaA))
    print("B = {} +/- {} nm".format(BB[c_index[1]], deltaB))
    print("chi squared ({}-2 d.o.f) = {}".format(len(data["measured_nm"]), np.min(cc)))

    if fn is not None:
        plt.savefig(fn+".pdf", bbox_inches='tight')