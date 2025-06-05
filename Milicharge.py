import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# DM parameters
eps = 2.2e-8  # in units of the electron charge
m_x = 5.0     # DM mass in GeV

#SI cross section
sig = 1e-46

# Recoil energy list
E_list = np.logspace(-0.5, 2, 100)

pl.figure(figsize=(7, 5))

# Xenon
pl.loglog(E_list, DMU.dRdE_millicharge(E_list, m_x, eps, "Xe131"), lw=1.5, label='Millicharge DM (Xe)')

# Argon
pl.loglog(E_list, DMU.dRdE_millicharge(E_list, m_x, eps, "Ar40"), lw=1.5, label='Millicharge DM (Ar)')

# C3F8 (3 Carbon + 8 Fluorine)
rate_c = DMU.dRdE_millicharge(E_list, m_x, eps, "C12")
rate_f = DMU.dRdE_millicharge(E_list, m_x, eps, "F19")
rate_c3f8 = 3 * rate_c + 8 * rate_f
pl.loglog(E_list, rate_c3f8, lw=1.5, label='Millicharge DM (C₃F₈)')

# Labels and grid
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
pl.savefig("plot plot.png")
pl.show()
