import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# --- Constants ---
c_A = 3.6e-8      # Anapole moment [GeV^-2]
m_x = 5.0         # Dark Matter mass [GeV]
alpha = 0.007297
e = np.sqrt(4 * np.pi * alpha)
gp = 5.59
gn = -3.83

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

# --- Spin-corrected Anapole Recoil Rate ---
def dRdE_anapole_spin_corrected(E, m_x, c_A, target):
    cp = [np.zeros_like(E) for _ in range(20)]
    cn = [np.zeros_like(E) for _ in range(20)]

    # O₈: velocity-dependent charge interaction — always applies
    cp[7] = -2.0 * e * c_A

    # O₉: spin-dependent magnetic interaction
    if target == "F19":  # unpaired proton
        cp[8] = -gp * c_A
        cn[8] = 0.0
    elif target in ["Xe129", "Xe131"]:  # unpaired neutron
        cp[8] = 0.0
        cn[8] = -gn * c_A
    else:  # spin-0 nuclei
        cp[8] = 0.0
        cn[8] = 0.0

    return DMU.dRdE_NREFT(E, m_x, cp, cn, target)

# --- Plotting ---
pl.figure(figsize=(7, 5))

# Xenon natural: Xe129 (45%), Xe131 (55%)
frac_xe129 = 0.45
frac_xe131 = 0.55
rate_xe129 = dRdE_anapole_spin_corrected(E_list, m_x, c_A, "Xe129")
rate_xe131 = dRdE_anapole_spin_corrected(E_list, m_x, c_A, "Xe131")
rate_xe = frac_xe129 * rate_xe129 + frac_xe131 * rate_xe131
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon (Natural)', color='blue')

# Argon (spin-0): includes O₈ only
rate_ar = dRdE_anapole_spin_corrected(E_list, m_x, c_A, "Ar40")
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')

# C₃F₈ = 3 C12 (spin-0, only O₈) + 8 F19 (O₈ + O₉)
mass_c = 0.1916
mass_f = 0.8084
rate_c = dRdE_anapole_spin_corrected(E_list, m_x, c_A, "C12") * mass_c
rate_f = dRdE_anapole_spin_corrected(E_list, m_x, c_A, "F19") * mass_f
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
ax.set_ylim(1e-11, None)
ax.set_xlim(0.5, None)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.title("Anapole Interaction with Spin-Corrected Operators")
pl.tight_layout()
pl.savefig("anapolemoment_spin_corrected_full.png")
pl.show()
