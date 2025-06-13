import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# --- Nuclear mass numbers ---
Avals = {
    "Xe131": 131,
    "Xe129": 129,
    "Ar40": 40,
    "C12": 12,
    "F19": 19
}

def dRdE_electric(E, m_x, d_x, target):
    """
    Return recoil rate for electric dipole DM interaction (via O11 only).
    O11 ~ 2ed / q^2, for both protons and neutrons.
    """
    A = Avals[target]
    amu = 931.5e3  # keV
    q = np.sqrt(2 * A * amu * E)  # momentum transfer in keV
    q2 = (q * 1e-6)**2  # in GeV²

    alpha = 0.007297
    e = np.sqrt(4 * np.pi * alpha)

    cp = [E * 0.0 for _ in range(20)]
    cn = [E * 0.0 for _ in range(20)]

    # 𝒪₁₁: EDM operator ~ 2ed/q² on both p and n
    cp[10] = 2.0 * e * d_x / q2
    cn[10] = 2.0 * e * d_x / q2

    return DMU.dRdE_NREFT(E, m_x, cp, cn, target)

# --- Parameters ---
d_x = 2.8e-8  # GeV^-1 (Electric dipole moment)
m_x = 5.0     # GeV (DM mass)
E_list = np.logspace(-0.5, 1.2, 100)

# --- Plot Setup ---
pl.figure(figsize=(7, 5))

# --- Xenon: average over Xe129 and Xe131
rate_xe129 = dRdE_electric(E_list, m_x, d_x, "Xe129")
rate_xe131 = dRdE_electric(E_list, m_x, d_x, "Xe131")
rate_xe = 0.45 * rate_xe129 + 0.55 * rate_xe131
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon', color='blue')

# --- Argon: spin-0, should contribute very little or nothing
rate_ar = dRdE_electric(E_list, m_x, d_x, "Ar40")
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')

# --- C₃F₈ = 8 F19 (active), 3 C12 (spin-0)
rate_f = dRdE_electric(E_list, m_x, d_x, "F19") * 0.8084
rate_c = dRdE_electric(E_list, m_x, d_x, "C12") * 0.1916  # mostly negligible
rate_c3f8 = rate_f + rate_c
pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

# --- Formatting ---
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.title("Electric Dipole Interaction via $𝒪_{11}$", loc='left')
pl.legend(loc='best')
pl.grid(True, which="both", ls="--", lw=0.5)

pl.tick_params(axis='both', which='both', direction='in', top=True, right=True)
pl.minorticks_on()
pl.tick_params(axis='both', which='minor', length=4, color='gray')
pl.tick_params(axis='both', which='major', length=7)

ax = pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-4, None)
ax.set_xlim(0.5, None)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.tight_layout()
pl.savefig("electric_dipole_O11_full.png")
pl.show()
