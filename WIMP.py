import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# DM parameters
g_a = 3.6e-8  # Anapole moment coupling [GeV^-2]
m_x = 5.0     # DM mass in GeV

# Recoil energy range [keV]
E_list = np.logspace(-0.5, 1.2, 100)

# --- C3F8 composition ---
mass_c = 0.1916
mass_f = 0.8084

# --- Compute differential rates ---
rate_xe = DMU.dRdE_anapole(E_list, m_x, g_a, "Xe131")
rate_ar = DMU.dRdE_anapole(E_list, m_x, g_a, "Ar40")
rate_c = DMU.dRdE_anapole(E_list, m_x, g_a, "C12") * mass_c
rate_f = DMU.dRdE_anapole(E_list, m_x, g_a, "F19") * mass_f
rate_c3f8 = rate_c + rate_f

# --- Print differential rates ---
print("Energy [keV]  |  dR/dE (Xenon)  |  dR/dE (Argon)  |  dR/dE (C₃F₈)")
print("-" * 70)
for i in range(len(E_list)):
    print(f"{E_list[i]:.4f}        {rate_xe[i]:.4e}     {rate_ar[i]:.4e}     {rate_c3f8[i]:.4e}")

# --- Plotting ---
pl.figure(figsize=(7, 5))
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon', color='blue')
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')
pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

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
#ax.set_ylim(1e-8, None)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.title("Anapolemoment")
pl.tight_layout()
pl.savefig("anapolemoment.png")
pl.show()
