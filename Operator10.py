from WIMpy import DMUtils as DMU
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Plot settings
font = {'family': 'sans-serif', 'size': 18}
mpl.rc('font', **font)
mpl.rcParams.update({
    'xtick.major.size': 5, 'xtick.major.width': 1,
    'xtick.minor.size': 3, 'xtick.minor.width': 1,
    'ytick.major.size': 5, 'ytick.major.width': 1,
    'ytick.minor.size': 3, 'ytick.minor.width': 1,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True
})

# Targets and nuclei
targets = ["Xenon", "Argon", "Germanium", "C3F8"]
nuclei_vals = {
    "Xenon": ["Xe128", "Xe129", "Xe130", "Xe131", "Xe132", "Xe134", "Xe136"],
    "Argon": ["Ar40"],
    "Germanium": ["Ge70", "Ge72", "Ge73", "Ge74", "Ge76"],
    "C3F8": ["C12", "F19"]
}

# Load nuclear fractions
nuclei_path = os.path.join(os.path.dirname(DMU.__file__), "Nuclei.txt")
nuclei_list = np.loadtxt(nuclei_path, usecols=(0,), dtype=str)
frac_list = np.loadtxt(nuclei_path, usecols=(3,))
frac_vals = dict(zip(nuclei_list, frac_list))

# Common energy list and DM mass
E_list = np.linspace(0.001, 100, 100)
m_x = 50.0  # GeV

# Calculate normalized spectrum
def calcSpectrum(target, operator, spin):
    cp = np.zeros(20)
    cn = np.zeros(20)
    cp[operator - 1] = 1.0
    cn[operator - 1] = 1.0
    dRdE = np.zeros_like(E_list)

    if target == "C3F8":
        dRdE = 0.1915 * DMU.dRdE_NREFT(E_list, m_x, cp, cn, "C12", j_x=spin) + \
               0.8085 * DMU.dRdE_NREFT(E_list, m_x, cp, cn, "F19", j_x=spin)
    else:
        for nuc in nuclei_vals[target]:
            dRdE += frac_vals[nuc] * DMU.dRdE_NREFT(E_list, m_x, cp, cn, nuc, j_x=spin)

    return dRdE

# Plot normalized spectra
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['r', 'b', 'g', 'c']

for target, color in zip(targets, colors):
    dRdE = calcSpectrum(target, operator=10, spin=0)
    tot = np.trapezoid(dRdE, E_list)
    dRdE_norm = dRdE / tot if tot != 0 else dRdE
    ax.plot(E_list, dRdE_norm, label=target, color=color, lw=1.5)

ax.set_xlabel(r'$E_R \,[\mathrm{keV}]$')
ax.set_ylabel(r'$\frac{dR}{dE_R} \, / \, \int \! dR/dE_R \, dE_R$')
ax.set_title(r"Normalized $\frac{dR}{dE_R}$ for Operator $\mathcal{O}_{10}$")
ax.set_ylim(0, 0.06)
ax.legend(fancybox=True, fontsize=12)
plt.tight_layout()
os.makedirs("../plots", exist_ok=True)
plt.savefig(f"../plots/Spectrum_Operator10_Spin0_mx={int(m_x)}GeV.pdf", bbox_inches="tight")
plt.show()

# --- Exclusion Cross Section vs DM Mass ---
threshold = 2.45  # keV
exposure = 1404  # kg-day
operator = 10
spin = 0
m_x_vals = np.logspace(0, 3, 200)  # 1 to 1000 GeV
sigma_all_targets = {}

for target in targets:
    sigma_vals = []
    for m_x in m_x_vals:
        try:
            cp = np.zeros(20)
            cn = np.zeros(20)
            cp[operator - 1] = 1.0
            cn[operator - 1] = 1.0
            E_list_local = np.linspace(threshold, 100.0, 500)

            if target == "C3F8":
                dRdE = (
                    0.1915 * DMU.dRdE_NREFT(E_list_local, m_x, cp, cn, "C12", j_x=spin) +
                    0.8085 * DMU.dRdE_NREFT(E_list_local, m_x, cp, cn, "F19", j_x=spin)
                )
            else:
                dRdE = np.zeros_like(E_list_local)
                for nuc in nuclei_vals[target]:
                    dRdE += frac_vals[nuc] * DMU.dRdE_NREFT(E_list_local, m_x, cp, cn, nuc, j_x=spin)

            R_exp = np.trapz(dRdE, E_list_local)
            sigma = 2.3 / (R_exp * exposure) if R_exp > 1e-10 else np.nan
            sigma_vals.append(sigma)
        except Exception as e:
            print(f"Error for {target} at m_x = {m_x:.2f} GeV: {e}")
            sigma_vals.append(np.nan)

    sigma_all_targets[target] = sigma_vals

# Plot sigma vs mass
plt.figure(figsize=(8, 6))
for target in targets:
    plt.loglog(m_x_vals, sigma_all_targets[target], label=target, linewidth=2)
    print(f"Sigma values for {target}:\n", sigma_all_targets[target])

plt.xlabel("DM Mass $m_\\chi$ [GeV]")
plt.ylabel(r"$\sigma^{\mathrm{SI}}_{\chi n,90}$ [pb]")
plt.title("Exclusion Cross Section vs DM Mass (Operator $\\mathcal{O}_{10}$)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray')
plt.tick_params(axis='both', which='major', length=7)
plt.savefig("../plots/Sigma_vs_Mass_AllTargets_Operator10.png", dpi=300)
plt.show()
