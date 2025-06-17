# coding: utf-8
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import LogLocator

import WIMpy.DMUtils as DMU
import WIMpy.WM as WM
import WIMpy.LabFuncs as LabFuncs

# --- DM and halo parameters ---
eps = 2.2e-8        # Millicharge (in units of e)
m_x = 5.0           # DM mass in GeV
rho0 = 0.3          # GeV/cm^3
vlag = 232.0        # km/s
sigmav = 156.0      # km/s
vesc = 544.0        # km/s

# --- Nuclear info ---
Jvals = {
    "Xe131": 1.5,
    "Ar40": 0.0,
    "C12": 0.0,
    "F19": 0.5
}

Avals = {
    "Xe131": 131,
    "Ar40": 40,
    "C12": 12,
    "F19": 19
}

# Define vmin

def vmin(E, A, m_x):
    m_A = A*0.9315
    mu = (m_A*m_x)/(m_A+m_x)
    v =  3e5*np.sqrt((E/1e6)*(m_A)/(2*mu*mu))
    return v
# --- Recoil rate for millicharged DM via O1 ---
def dRdE_millicharge(E, m_x, epsilon, target):
    A = Avals[target]
    J = Jvals[target]
    amu = 931.5e3  # keV

    q1 = np.sqrt(2 * A * amu * E)          # momentum in keV
    q2 = q1 * (1e-12 / 1.97e-7)            # q in 1/GeV
    b = np.sqrt(41.467 / (45*A**(-1/3) - 25*A**(-2/3)))
    y = (q2 * b / 2)**2

    v_min = vmin(E, A, m_x)
    eta = DMU.calcEta(v_min, vlag=vlag, sigmav=sigmav, vesc=vesc)

    alpha = 0.007297
    e = np.sqrt(4 * np.pi * alpha)
    cp = 2*epsilon * e**2
    cn = 0.0

    rate = E * 0.0
    couplings = [cp + cn, cp - cn]

    for tau1 in [0, 1]:
        for tau2 in [0, 1]:
            c18 = couplings[tau1]
            c19 = couplings[tau2]
            R_M = c18 * c19 * eta / (q1 * 1e-6)**4  # convert q to GeV
            rate += R_M * np.vectorize(WM.calcwm)(tau1, tau2, y, target)

    conv = (rho0 / (2 * np.pi * m_x)) * 1.69612985e14  # to dR/dE in keV⁻¹ kg⁻¹ day⁻¹
    rate = np.clip(rate, 0, np.inf)
    return (4*np.pi/(2*Jvals[target]+1))*rate*conv
# --- Recoil energy range ---
E_list = np.logspace(-0.5, 1.2, 100)

# --- Compute rates ---
rate_xe = dRdE_millicharge(E_list, m_x, eps, "Xe131")
rate_ar = dRdE_millicharge(E_list, m_x, eps, "Ar40")
rate_f = dRdE_millicharge(E_list, m_x, eps, "F19") * 0.8084
rate_c = dRdE_millicharge(E_list, m_x, eps, "C12") * 0.1916
rate_c3f8 = rate_f + rate_c

# --- Plot ---
pl.figure(figsize=(7, 5))
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon', color='blue')
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')
pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

# --- Formatting ---
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.title("Millicharged DM")
pl.legend(loc='best')
pl.grid(True, which="both", ls="--", lw=0.5)
pl.tick_params(axis='both', which='both', direction='in', top=True, right=True)
pl.minorticks_on()
pl.tick_params(axis='both', which='minor', length=4, color='gray')
pl.tick_params(axis='both', which='major', length=7)

ax = pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(None, None)
ax.set_xlim(None, None)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.tight_layout()
pl.savefig("millicharge_O1_scaled.png")
pl.show()
