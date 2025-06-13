import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

# DM parameters
g_a = 3.6e-8  # Anapole moment coupling [GeV^-2]
m_x = 5.0     # DM mass in GeV

# Recoil energy range [keV]
E_list = np.logspace(-0.5, 1.2, 100)

# --- Function definition ---
def dRdE_anapole(E, m_x, c_A, target):
    """Return recoil rate for anapole Dark Matter.
    Parameters
    ----------
    * `E` [array]: Recoil energies.
    * `m_x` [float]: Dark Matter mass in GeV.
    * `c_A` [float]: Dark Matter anapole moment (in GeV^-2).
    * `target` [string]: Recoil target.
    * `vlag` [float] (optional): Average lag speed of the lab in km/s.
    * `sigmav` [float] (optional): Velocity dispersion of the DM halo in km/s.
    * `vesc` [float] (optional): Escape speed in the Galactic frame in km/s.
    Returns
    -------
    * `rate` [array like]: Recoil rate in units of events/keV/kg/day.
    """
    # See https://arxiv.org/pdf/1401.4508.pdf
    alpha = 0.007297
    e = np.sqrt(4*np.pi*alpha)
    gp = 5.59
    gn = -3.83

    cn = np.zeros(20)
    cp = np.zeros(20)

    # Operator 8
    cp[7] = -2 * e * c_A
    

    # Operator 9
    cp[8] = -gp * c_A
    cn[8] = -gn * c_A

    return DMU.dRdE_NREFT(E, m_x, cp, cn, target)

# --- Plotting ---
pl.figure(figsize=(7, 5))

# --- Xenon ---
rate_xe = dRdE_anapole(E_list, m_x, g_a, "Xe131")
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon', color='blue')

# --- Argon ---
rate_ar = dRdE_anapole(E_list, m_x, g_a, "Ar40")
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')

# --- C3F8 ---
mass_c = 0.1916
mass_f = 0.8084
rate_c = dRdE_anapole(E_list, m_x, g_a, "C12") * mass_c 
rate_f = dRdE_anapole(E_list, m_x, g_a, "F19") * mass_f 
rate_c3f8 = rate_c + rate_f
pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

# --- Labels and formatting ---
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.legend(loc='best')
pl.grid(True, which="both", ls="--", lw=0.5)

# --- Axis ticks and limits ---
pl.tick_params(axis='both', which='both', direction='in', top=True, right=True)
pl.minorticks_on()
pl.tick_params(axis='both', which='minor', length=4, color='gray')
pl.tick_params(axis='both', which='major', length=7)

ax = pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(10 * 1e-12, None)
ax.set_xlim(0.5, None)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.title("anapolemoment")
pl.tight_layout()
pl.savefig("anapolemoment")
pl.show()
