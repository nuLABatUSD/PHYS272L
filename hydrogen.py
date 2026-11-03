import numpy as np
import matplotlib.pyplot as plt

def make_calibration_function(spec_num):
    spec_file = np.load("PHYS272L/spectrometer_calibration.npz", allow_pickle=True)
    
    works = False
    for key in spec_file.files:
        if spec_num == key[-4:]:
            works = True

    if not works:
        print("Spectrometer not found. Please double check.")
        print(spec_file.files)
        return None

    cal = spec_file["spec"+spec_num].item()
    ABC_vals = cal["ABC_vals"]
    L_mat = cal["L_mat"]

    def func(meas, dmeas):
        sv = (meas-550)/1000

        values = np.zeros((2, ABC_vals.shape[1], ABC_vals.shape[2], ABC_vals.shape[3]))
        for i in range(ABC_vals.shape[1]):
            for j in range(ABC_vals.shape[2]):
                for k in range(ABC_vals.shape[3]):
                    p = np.poly1d(ABC_vals[:,i,j,k])
                    values[0,i,j,k] = p(sv)
                    values[1,i,j,k] = p(sv)**2
        
        ev = np.sum(values[0,:,:,:] * L_mat) / np.sum(L_mat)
        var = np.sum(values[1,:,:,:] * L_mat) / np.sum(L_mat)
        sd = np.sqrt(var - ev**2)

        return meas + ev, np.sqrt(sd**2+dmeas**2)

    return func

def print_calibrated_wavelengths(calib, measured_values):
    for i in range(len(measured_values)):
        print(calib(measured_values[i][0], measured_values[i][1]))

def make_data_dictionary(wv):
    n = np.zeros(len(wv))
    lam = np.zeros_like(n)
    dlam = np.zeros_like(n)

    for i in range(len(wv)):
        n[i], lam[i], dlam[i] = wv[i]

    data = dict()
    data['n'] = n
    data['lam'] = lam
    data['dlam'] = dlam

    return data

def chi_sq(data, R):
    Egamma = hc / data['lam']
    m_val = np.min(data['n'])-1

    cc = (Egamma - R*(1/m_val**2 - 1/data['n']**2))**2
    cc /= ( hc / data['lam']**2 * data['dlam'])**2

    return np.sum(cc)

hc = 1239.84
Ry = 13.6057
def best_fit(data, fn=None):
    R_exp = 13.6
    R_mult = 0.5
    for k in range(10):
        RR = np.linspace(R_exp*(1-R_mult), R_exp*(1+R_mult), 100)
        cc = np.zeros(len(RR))
        for i in range(len(RR)):
                cc[i] = chi_sq(data, RR[i])

        R_best = RR[np.argmin(cc)]
        R_ind = np.where(cc < np.min(cc)+1)[0]

        if len(R_ind) < 10:
            R_mult *= 0.5
        else:
            break
        R_exp = R_best

    print("Best Fit:")
    print("R = {} +/- {} eV".format(R_best, 0.5*(RR[R_ind[-1]]-RR[R_ind[0]])))
    print("chi-squared ({}-1 d.o.f) = {}".format(len(data['n']), np.min(cc)))
    print("====================")
    print("chi-squared (R = 1 Ry, {} d.o.f) = {}".format(len(data['n']), chi_sq(data, Ry)))

    plt.figure(figsize=(4,3))

    nn = np.linspace(np.min(data['n']), np.max(data['n']))
    plt.plot(nn, hc/R_best*(1/(np.min(data['n'])-1)**2-1/nn**2)**-1)

    plt.errorbar(data['n'], data['lam'],  yerr=data['dlam'], fmt='o', ms=3, barsabove=True, capsize=2, elinewidth=1, color='k')

    plt.xlabel("n", fontsize=13)
    plt.ylabel("wavelength [nm]", fontsize=13)

    if fn is not None:
        plt.savefig(fn+".pdf", bbox_inches='tight')