import numpy as np
import matplotlib.pyplot as pl
import WIMpy.DMUtils as DMU
from matplotlib.ticker import LogLocator

#Number of protons and neutrons in Xenon
A_Xe = 131.
N_p_Xe = 54.
N_n_Xe = 131 - N_p_Xe

E_list = np.logspace(-3, 2,100)

mu_x = 1e-8 # in units of the Bohr Magneton
eps = 1e-10 #in units of the electron charge
c_A = 1e-5 #in units of GeV^-2

#DM mass in GeV
m_x = 100.0



pl.figure(figsize=(7,5))
pl.loglog(E_list, DMU.dRdE_standard(E_list, N_p_Xe, N_n_Xe, m_x, sig), lw=1.5, label='Spin-independent')
pl.loglog(E_list, DMU.dRdE_millicharge(E_list, m_x, eps, "Xe131"), lw=1.5, label='Millicharge DM')
pl.loglog(E_list, DMU.dRdE_magnetic(E_list, m_x, mu_x, "Xe131"), lw=1.5, label='Magnetic DM')
pl.loglog(E_list, DMU.dRdE_anapole(E_list, m_x, c_A, "Xe131"), lw=1.5, label='Anapole DM')

pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.legend(loc = 'best')
pl.show()
