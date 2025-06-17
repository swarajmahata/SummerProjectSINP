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

def dRdE_magnetic(E, m_x, mu_x, target,j_x=1):
    
    
    A = Avals[target]
    
    #See Eq. 62 of https://arxiv.org/pdf/1307.5955.pdf, but note
    #that we're using some different normalisations for the operators
    #so there are some extra factors of m_x and m_p lurking around...
    
    amu = 931.5e3 # keV
    q1 = np.sqrt(2*A*amu*E) #Recoil momentum in keV
    
    alpha = 0.007297
    e = np.sqrt(4*np.pi*alpha)
    m_p = 0.9315
    
    #Proton and neutron g-factors
    gp = 5.59
    gn = -3.83
    
    #Bohr Magneton
    #Tesla   = 194.6*eV**2           # Tesla in natural units (with e = sqrt(4 pi alpha))
    #muB     = 5.7883818e-5*eV/Tesla # Bohr magneton
    mu_B = 297.45 #GeV^-1 (in natural units (with e = sqrt(4 pi alpha)))
    cp = [E*0.0 for i in range(20)]
    cn = [E*0.0 for i in range(20)] 
    
   #operator 19  
    cp[18] = 2.0 * e * mu_x / q1
    cn[18] = 2.0 * e * mu_x / q1
   #opefrator 20
    cp[19] = 2.0 * e * mu_x / q1
    cn[19] = 2.0 * e * mu_x / q1

    return DMU.dRdE_NREFT(E, m_x, cp, cn, target,j_x=1)

# --- Parameters ---
mu_x = 2.8e-8
m_x = 5.0
E_list = np.logspace(-0.5, 1.2, 100)

pl.figure(figsize=(7, 5))

# --- Xenon: unpaired neutron in Xe129/Xe131
rate_xe129 =  dRdE_magnetic(E_list, m_x, mu_x, "Xe129")
rate_xe131 =  dRdE_magnetic(E_list, m_x, mu_x, "Xe131")
rate_xe = 0.45 * rate_xe129 + 0.55 * rate_xe131
pl.loglog(E_list, rate_xe, lw=2, ls='--', label='Xenon', color='blue')

# --- Argon: spin-0 nucleus (Ar40)
rate_ar = dRdE_magnetic(E_list, m_x, mu_x, "Ar40")
pl.loglog(E_list, rate_ar, lw=2, ls=':', label='Argon', color='green')

# --- C3F8 = 3 C12 (spin-0) + 8 F19 (unpaired proton)
rate_f =  dRdE_magnetic(E_list, m_x, mu_x, "F19") * 0.8084
rate_c = np.zeros_like(E_list)
rate_c3f8 = rate_c + rate_f
pl.loglog(E_list, rate_c3f8, lw=2, label='C₃F₈', color='red')

# --- Formatting ---
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.title("Magnetic Dipole Interaction with Spin-Corrected Operators")
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
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)
ax.tick_params(axis='y', which='major', length=6, width=1.2)

pl.tight_layout()
pl.savefig("magneticmoment_spin_corrected_final.png")
pl.show()
