import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# DM parameters
mu_x = 2.8e-8  # Magnetic dipole moment coupling [GeV^-1]
m_x = 5.0      # DM mass in GeV

# Recoil energy range [keV]
E_list = np.logspace(-0.5, 1.2, 100)

pl.figure(figsize=(7, 5))

# Xenon
rate_xe = DMU.dRdE_magnetic(E_list, m_x, mu_x, "Xe131")
rate_xe[rate_xe < 1e0] = np.nan
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon', color='blue')

# Argon
rate_ar = DMU.dRdE_magnetic(E_list, m_x, mu_x, "Ar40")
rate_ar[rate_ar < 1e0] = np.nan
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')

# Mass fractions for 1 kg of C3F8
mass_c = 0.1916  # kg
mass_f = 0.8084  # kg

# C3F8 = 3 C12 + 8 F19
rate_c = DMU.dRdE_magnetic(E_list, m_x, mu_x, "C12") * mass_c
rate_f = DMU.dRdE_magnetic(E_list, m_x, mu_x, "F19") * mass_f
rate_c3f8 = rate_c + rate_f
rate_c3f8[rate_c3f8 < 1e0] = np.nan

pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

# Labels and formatting
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.legend(loc='best')
pl.grid(True, which="both", ls="--", lw=0.5)

# --- Axis ticks ---
pl.tick_params(axis='both', which='both', direction='in', top=True, right=True)
pl.minorticks_on()
pl.tick_params(axis='both', which='minor', length=4, color='gray')
pl.tick_params(axis='both', which='major', length=7)

# Force minor ticks on log y-axis
ax = pl.gca()
ax.set_yscale('log')
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)
pl.title("Magneticdipolemoment")
pl.savefig("Magneticdipolemoment.png")
pl.show()