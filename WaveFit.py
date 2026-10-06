import numpy as np
import matplotlib.pyplot as plt

def make_data_dictionary(freq_data, mass_in_g, L_in_cm):
    if len(mass_in_g) != len(freq_data):
        print("mass_in_g needs to have the same number of values as there are frequency data")
        return None
    if len(L_in_cm) != 2:
        print("L_in_cm needs to be the Length in cm, and be a list of two elements: value and uncertainty")
        return None
    data = dict()

    mass = np.array(mass_in_g)
    data["mass"] =  mass

    freq = []
    N = 0
    for f in freq_data:
        freq.append(np.array(f))
        N += len(f)
    data["frequency"] = freq
    data["N_data"] = N

    data["L_cm"] = L_in_cm[0]
    data["dL"] = L_in_cm[1]

    v_est = np.zeros(len(mass_in_g))
    for i in range(len(v_est)):
        v_est[i] = np.sum(freq[i][:,2] * freq[i][:,0] / freq[i][:,1]**2) / np.sum(freq[i][:,2]**2 / freq[i][:,1]**2)

    fig, ax = plt.subplots(figsize=(10,6), nrows=2, ncols=2, sharey=True, sharex=True)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    nn = np.linspace(0, 6.5)
    for i in range(4):
        ax[i//2][i%2].errorbar(freq[i][:,0], freq[i][:,2], xerr=freq[i][:,1], fmt='o', capsize=3, ms=3)

        ax[i//2][i%2].plot( nn*v_est[i], nn)
        ax[i//2][i%2].text(0, 6, "{} g; v = {:.1f} m/s".format(mass[i], v_est[i] * (2 * L_in_cm[0] / 100.)), ha='left')

    for i in range(2):
        ax[1][i].set_xlabel("Frequency [Hz]")
        ax[i][0].set_ylabel("n")

    data["v_est"] = v_est * (2 * L_in_cm[0] / 100.)

    return data

mu_g__m = 1.30
def chi_sq(data, g):
    chi = 0
    for i in range(4):
        v = np.sqrt(data["mass"][i] * g / mu_g__m)
        cc = (data["frequency"][i][:,0] - v / (2 * data["L_cm"] / 100.) * data["frequency"][i][:,2])**2
        cc /= data["frequency"][i][:,1]**2
        chi += np.sum(cc)
    return chi

def find_best_value(data, cc, gg):
    min_index = np.argmin(cc)
    sig_range = np.where(cc < np.min(cc)+1)[0]

    g_val = None
    if min_index == 0:
        g_val = np.linspace( gg[0] - (gg[-1]-gg[0]), gg[0], 100)
    elif min_index == len(cc)-1:
        g_val = np.linspace( gg[-1], gg[-1] + (gg[-1] - gg[0]), 100)
    elif len(sig_range) == 1:
        g_val = np.linspace( gg[sig_range[0]-1], gg[sig_range[0]+1], 100)
    elif len(sig_range) == len(cc):
        g_val = np.linspace( gg[0] - (gg[-1]-gg[0]), gg[-1] + (gg[-1]-gg[0]), 100)
    elif sig_range[0] == 0 and sig_range[-1] != len(cc)-1:
        g_val = np.linspace( gg[0] - (gg[sig_range[-1]+1]-gg[0]), gg[sig_range[-1]+1], 100)
    elif sig_range[-1] == len(cc)-1 and sig_range[0] != 0:
        g_val = np.linspace( gg[sig_range[0]-1], gg[-1] + (gg[-1] - gg[sig_range[0]-1]), 100)
    elif len(sig_range) < 10:
        g_val = np.linspace( gg[sig_range[0]-1], gg[sig_range[-1]+1], 50)

    if g_val is not None:
        c_val = np.zeros(len(g_val))
        for i in range(len(c_val)):
            c_val[i] = chi_sq(data, g_val[i])
        return find_best_value(data, c_val, g_val)
    else:
        return gg[min_index], 0.5*np.abs(gg[sig_range[-1]] - gg[sig_range[0]]), np.min(cc)


def fit_g(data, filename=None):
    gg = np.linspace(9.6, 10, 101)
    cc = np.zeros(len(gg))
    for i in range(len(cc)):
        cc[i] = chi_sq(data, gg[i])

    g_val, dg_val, cmin = find_best_value(data, cc, gg)
    print("Best fit")
    print("g = {} +/- {} m/s/s".format(g_val, dg_val))
    print("chi-squared ({}-1 d.o.f) = {}".format(data["N_data"], cmin))
    #return (find_best_value(data, cc, gg))

    plt.figure(figsize=(5,3))
    plt.scatter(data["mass"], data["v_est"])

    plt.xlim(0, 1.05 * np.max(data["mass"]))
    plt.ylim(0, 1.05 * np.max(data["v_est"]))
    xx = np.linspace(0, np.max(data["mass"]), 1000)
    plt.plot(xx, np.sqrt(xx * g_val / mu_g__m))
    plt.xlabel("mass [g]", fontsize=14)
    plt.ylabel("wave speed [m/s]", fontsize=14)
    plt.tick_params(labelsize=12)

    if filename is not None:
        plt.savefig(filename + ".pdf", bbox_inches='tight')