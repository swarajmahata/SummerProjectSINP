import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# --- Constants ---
mu_x = 2.8e-8     # Magnetic dipole moment [GeV^-1]
m_x = 5.0         # Dark Matter mass [GeV]
alpha = 0.007297
e = np.sqrt(4 * np.pi * alpha)
gp = 5.59
gn = -3.83
mu_B = 297.45     # Bohr magneton [GeV^-1]
m_p = 0.9315      # Proton mass [GeV]

# --- Target Mass Numbers ---
Avals = {
    "Xe131": 131,
    "Xe129": 129,
    "Ar40": 40,
    "C12": 12,
    "F19": 19
}

# --- Recoil energy range [keV] ---
E_list = np.logspace(-0.5, 1.2, 100)

# --- Spin-corrected Magnetic Dipole Recoil Rate ---
def dRdE_magnetic_spin_corrected(E, m_x, mu_x, target):
    cp = [np.zeros_like(E) for _ in range(20)]
    cn = [np.zeros_like(E) for _ in range(20)]

    A = Avals[target]
    amu = 931.5e3  # keV
    q = np.sqrt(2 * A * amu * E)  # keV
    q_GeV = q * 1e-6  # Convert to GeV

    # O₁: Dipole-charge interaction (target independent)
    cp[0] = e * mu_x * mu_B / (2.0 * m_x)

    if target == "F19":  # unpaired proton
        cp[3] = gp * e * mu_x * mu_B / m_p                   # O₄
        cp[4] = 2 * e * mu_x * mu_B * m_p / q_GeV**2         # O₅
        cp[5] = -gp * e * mu_x * mu_B * m_p / q_GeV**2       # O₆
    elif target in ["Xe129", "Xe131"]:  # unpaired neutron
        cn[3] = gn * e * mu_x * mu_B / m_p
        cp[4] = 2 * e * mu_x * mu_B * m_p / q_GeV**2
        cn[5] = -gn * e * mu_x * mu_B * m_p / q_GeV**2
    else:
        pass  # spin-0 nuclei, no spin interaction

    return DMU.dRdE_NREFT(E, m_x, cp, cn, target)

# --- Plotting ---
pl.figure(figsize=(7, 5))

# Xenon natural: Xe129 (45%), Xe131 (55%)
frac_xe129 = 0.45
frac_xe131 = 0.55
rate_xe129 = dRdE_magnetic_spin_corrected(E_list, m_x, mu_x, "Xe129")
rate_xe131 = dRdE_magnetic_spin_corrected(E_list, m_x, mu_x, "Xe131")
rate_xe = frac_xe129 * rate_xe129 + frac_xe131 * rate_xe131
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon (Natural)', color='blue')

# Argon (spin-0)
rate_ar = dRdE_magnetic_spin_corrected(E_list, m_x, mu_x , "Ar40")
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')

# C₃F₈ = 3 C12 (spin-0) + 8 F19
mass_c = 0.1916
mass_f = 0.8084
rate_c = np.zeros_like(E_list)  # C12 has spin-0
rate_f = dRdE_magnetic_spin_corrected(E_list, m_x, mu_x, "F19") * mass_f
rate_c3f8 = rate_c + rate_f
pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

# --- Formatting ---
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.legend(loc='best')
pl.grid(True, which="both", ls="--", lw=0.5)
pl.tick_params(axis='both', which='both', direction='in', top=True, right=True)
pl.minorticks_on()
pl.tick_params(axis='both', which='minor', length=4, color='gray')
pl.tick_params(axis='both', which='major', length=7)

ax = pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(2 * 1e-4, None)
ax.set_xlim(0.5, None)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.title("Magnetic Dipole Interaction with Spin-Corrected Operators")
pl.tight_layout()
pl.savefig("magneticmoment_spin_corrected_full.png")
pl.show()
