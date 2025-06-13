import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# --- Constants ---
g_a = 3.6e-8     # Anapole moment coupling [GeV^-2]
m_x = 5.0        # DM mass [GeV]
alpha = 0.007297
e = np.sqrt(4 * np.pi * alpha)
gp = 5.59
gn = -3.83

# --- Recoil energy range [keV] ---
E_list = np.logspace(-0.5, 1.2, 100)

def dRdE_anapole_spin_corrected(E, m_x, c_A, target):
    """Recoil rate with correct nuclear spin weighting for anapole interactions."""
    cp = np.zeros(20)
    cn = np.zeros(20)

    # 𝒪₈: velocity-dependent charge operator
    cp[7] = -2.0 * e * c_A

    # 𝒪₉: spin-dependent magnetic operator
    if target in ["F19"]:  # unpaired proton
        cp[8] = -gp * c_A
        cn[8] = 0.0
    elif target in ["Xe129", "Xe131"]:     # unpaired neutron
        cp[8] = 0.0
        cn[8] = -gn * c_A 
    else:
        cp[8] = 0.0
        cn[8] = 0.0

    return DMU.dRdE_NREFT(E, m_x, cp, cn, target)

# --- Plot Setup ---
pl.figure(figsize=(7, 5))

# --- Xenon (natural abundance)
frac_xe129 = 0.45
frac_xe131 = 0.55
rate_xe129 = dRdE_anapole_spin_corrected(E_list, m_x, g_a, "Xe129") 
rate_xe131 = dRdE_anapole_spin_corrected(E_list, m_x, g_a, "Xe131") 
rate_xe = frac_xe129 * rate_xe129 + frac_xe131 * rate_xe131
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon (Natural)', color='blue')

# --- Argon (spin-0, no contribution) ---
rate_ar = dRdE_anapole_spin_corrected(E_list, m_x, g_a, "Ar40")
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')

# --- C₃F₈ (your original version)
mass_f = 0.8084
mass_c = 0.1916
rate_f = dRdE_anapole_spin_corrected(E_list, m_x, g_a, "F19") * mass_f
rate_c = np.zeros_like(E_list)
rate_c3f8 = rate_c + rate_f
pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

# --- Plot Formatting ---
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
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.title("Anapole Interaction with Spin-Corrected Operators")
pl.tight_layout()
pl.savefig("anapolemoment_spin_corrected_fixed_xenon")
pl.show()
