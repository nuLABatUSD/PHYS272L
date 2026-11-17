import numpy as np
import matplotlib.pyplot as plt

def make_data_dictionary(nrg):
    vv = np.zeros(len(nrg))
    en = np.zeros_like(vv)
    den = np.zeros_like(vv)

    for i in range(len(vv)):
        vv[i] = nrg[i][0]
        en[i] = nrg[i][1]
        den[i] = nrg[i][2]

    data = dict()
    data['v'] = vv
    data['energy'] = en
    data['denergy'] = den

    plt.figure(figsize=(4,3))
    plt.errorbar(data['v'], data['energy'], yerr=data['denergy'], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    plt.xlabel(r"$v''$", fontsize=12)
    plt.ylabel("E [eV]", fontsize=12)
    plt.show()
    return data

def quadr(x, A, B, C):
    return A*x**2 + B*x + C

def chi_sq(x, y, s, A, B, C):
    cc = y - quadr(x, A, B, C)
    cc = cc**2
    cc /= s**2
    return np.sum(cc)
    
def lin_fit(data):
    x = data['v']+0.5
    y = data['energy']
    s = data['denergy']

    A = np.sum(x / s**2)
    B = np.sum(1 / s**2)
    C = np.sum(y / s**2)
    D = np.sum(x**2 / s**2)
    E = np.sum(x*y / s**2)
    F = np.sum(y**2 / s**2)

    hw_best = (E*B-C*A)/(D*B-A**2)
    E0_best = (D*C-E*A)/(D*B-A**2)

    d_hw = np.sqrt(B/(D*B-A**2))
    d_E0 = np.sqrt(D/(D*B-A**2))

    return hw_best, d_hw, E0_best, d_E0, chi_sq(x, y, s, 0, hw_best, E0_best)

def quad_fit(data):
    x = data['v']+0.5
    y = data['energy']
    s = data['denergy']

    zz = np.polyfit(x, y, 2)

    b0 = zz[1]
    c0 = zz[2]
    a0 = zz[0] / b0

    AA = np.linspace(0.5*a0, 1.5*a0, 51)
    BB = np.linspace(0.75*b0, 1.25*b0, 51)
    
    cc = np.zeros((51, 51))

    for i in range(51):
        for j in range(51):
            c_mean = np.mean(y - AA[i]* BB[j]*x**2 - BB[j]*x)
            CC = np.linspace(0.75*c_mean, 1.25*c_mean, 25)
            vv = np.zeros(25)
            for k in range(25):
                vv[k] = chi_sq(x, y, s, AA[i]*BB[j], BB[j], CC[k])
            cc[i,j] = np.min(vv)
            #print(cc[i,j], AA[i], BB[j], CC[np.argmin(vv)])

    ai, bi = np.where(cc < np.min(cc)+25)
    AA = np.linspace(np.min(AA[ai]), np.max(AA[ai]), 51)
    BB = np.linspace(np.min(BB[bi]), np.max(BB[bi]), 51)

    cc = np.zeros((51, 51))

    for i in range(51):
        for j in range(51):
            c_mean = np.mean(y - AA[i]*BB[j]*x**2 - BB[j]*x)
            CC = np.linspace(0.75*c_mean, 1.25*c_mean, 25)
            vv = np.zeros(25)
            for k in range(25):
                vv[k] = chi_sq(x, y, s, AA[i]*BB[j], BB[j], CC[k])
            cc[i,j] = np.min(vv)

    ind = np.unravel_index(np.argmin(cc), cc.shape)

    bind = np.where(cc[ind[0],:] < np.min(cc)+1)[0]
    aind = np.where(cc[:,ind[1]] < np.min(cc)+1)[0]

    c_mean = np.mean(y - AA[ind[0]]*BB[ind[1]]*x**2 - BB[ind[1]]*x)
    CC = np.linspace(0.75*c_mean, 1.25*c_mean, 25)
    chi = np.zeros(25)
    for i in range(25):
        chi[i] = chi_sq(x, y, s, AA[ind[0]]*BB[ind[1]], BB[ind[1]], CC[i])

    return AA[ind[0]], 0.5*np.abs(AA[aind[-1]]-AA[aind[0]]), BB[ind[1]], 0.5*np.abs(BB[bind[-1]]-BB[bind[0]]), CC[np.argmin(chi)], np.min(cc)

def data_fit(data, file_name=None):
    lf = lin_fit(data)
    print("Harmonic Oscillator Fit")
    print("hbar omega_0 = {} +/- {} eV".format(lf[0], lf[1]))
    print("chi square ({}-2 d.o.f) = {}".format(len(data['v']), lf[-1]))
    print("=================")

    af = quad_fit(data)
    print("Anharmonic Oscillator Fit")
    print("hbar omega_0 = {} +/- {} eV".format(af[2], af[3]))
    print("x = {} +/- {}".format(af[0], af[1]))
    print("chi square ({}-3 d.o.f) = {}".format(len(data['v']), af[-1]))

    fig, ax = plt.subplots(figsize=(10,4),nrows=2, ncols=2, sharex=True, height_ratios=[2,1])
    plt.subplots_adjust(hspace=0)
    ax[0][0].set_title("Harmonic Oscillator", fontsize=11)
    ax[0][1].set_title("Anharmonic Oscillator", fontsize=11)

    ax[0][0].errorbar(data['v'], data['energy'], yerr=data['denergy'], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    ax[0][1].errorbar(data['v'], data['energy'], yerr=data['denergy'], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')

    xx = np.linspace(np.min(data['v']), np.max(data['v']), 100)
    ax[0][0].plot(xx, quadr(xx+0.5, 0, lf[0], lf[2]))
    ax[0][1].plot(xx, quadr(xx+0.5, af[0]*af[2], af[2], af[4]))
    ax[1][0].errorbar(data['v'], data['energy']-quadr(data['v']+0.5, 0, lf[0], lf[2]), yerr=data['denergy'], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')
    ax[1][1].errorbar(data['v'], data['energy']-quadr(data['v']+0.5, af[0]*af[2], af[2], af[4]), yerr=data['denergy'], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')

    ax[0][0].set_ylabel("E [eV]", fontsize=11)
    ax[1][0].set_ylabel("residual [eV]", fontsize=11)
    for i in range(2):
        ax[1][i].axhline(0, c='0.50', ls='--')
        ax[1][i].set_xlabel(r"$v''$", fontsize=11)
    if file_name is not None:
        plt.savefig(file_name+".png", bbox_inches='tight')