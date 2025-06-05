import WIMpy
from WIMpy import DMUtils as DMU
print(WIMpy.__version__)

# %matplotlib inline  # Uncomment only if running in Jupyter

# Import libraries
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl

# Matplotlib settings
font = {'family': 'sans-serif', 'size': 16}
mpl.rc('font', **font)
mpl.rcParams.update({
    'xtick.major.size': 5,
    'xtick.major.width': 1,
    'xtick.minor.size': 3,
    'xtick.minor.width': 1,
    'ytick.major.size': 5,
    'ytick.major.width': 1,
    'ytick.minor.size': 3,
    'ytick.minor.width': 1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

from tqdm import tqdm
from scipy.integrate import quad
from scipy.special import erf

# List available targets
print("Available targets:", DMU.target_list)

# Standard SI cross section calculation
cp = np.zeros(20)
cn = np.zeros(20)
cp[0] = 1e-9 / (246.2)**2
cn[0] = 1e-9 / (246.2)**2
print("Couplings to protons [GeV^{-2}]:", cp)
print("Couplings to neutrons [GeV^{-2}]:", cn)

m_x = 100  # DM mass in GeV

# Xenon target
A_Xe = 131.
N_p_Xe = 54.
N_n_Xe = A_Xe - N_p_Xe

# Convert coupling to cross-section
sig = DMU.coupling_to_xsec(cp[0], m_x)

E_list = np.logspace(-3, 2, 1000)

R_SI = DMU.dRdE_standard(E_list, N_p_Xe, N_n_Xe, m_x, sig)
R_SI_NREFT = DMU.dRdE_NREFT(E_list, m_x, cp, cn, target="Xe131")

pl.figure(figsize=(6, 4))
pl.plot(E_list, R_SI, label="Standard SI")
pl.plot(E_list, R_SI_NREFT, '--', lw=2.0, label=r'$\mathcal{O}_1$')
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.legend()
pl.title("Spin-Independent Scattering")
pl.savefig(" plot 1")
pl.grid()
pl.show()

#Spin-independent coupling
cp_SI = np.zeros(20)
cp_SI[0] = 2.5e-5

cn_SI = 1.0*cp_SI

#Spin-dependent coupling
cp_SD = np.zeros(20)
cp_SD[3] = 1.0

cn_SD = np.zeros(20)
cn_SD[3] = 2.5e-2

m_x = 100 #DM mass in GeV

E_list = np.logspace(-3, 2,1000)

pl.figure()

pl.plot(E_list, 1e-3*DMU.dRdE_NREFT(E_list, m_x, cp_SI, cn_SI, "Xe131"),'k--' , lw=1.5,label ='SI')
pl.plot(E_list, 1e-3*DMU.dRdE_NREFT(E_list, m_x, cp_SD, 0.0*cp_SD, "Xe131"),'r-' , lw=1.5,label ='SD (p)')
pl.plot(E_list, 1e-3*DMU.dRdE_NREFT(E_list, m_x, 0.0*cn_SD, cn_SD, "Xe131"),'b-' , lw=1.5,label ='SD (n)')
pl.xlabel(r'$E_R$ [keV]')
#pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
#pl.title('Random NREFT interactions')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [arb. units]')

pl.legend(loc='best',fancybox = True)
pl.savefig(" plot 2")
pl.show()

#Spin-independent coupling
cp_SI = np.zeros(20)
cp_SI[0] = 2.5e-5

cn_SI = 1.0*cp_SI

m_x = 10 #DM mass in GeV

E_list = np.logspace(-3, 2,1000)

pl.figure(figsize=(10,6))

#Loop over all available target
target_labs = DMU.target_list

colors = pl.cm.tab20(np.linspace(0,1,len(target_labs)))
for i, target in enumerate(target_labs):
    pl.loglog(E_list, 1e-3*DMU.dRdE_NREFT(E_list, m_x, cp_SI, cn_SI, target), lw=1.5,label =target, color=colors[i])

pl.xlabel(r'$E_R$ [keV]')
#pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
#pl.title('Random NREFT interactions')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [arb. units]')

pl.xlim(1e-3, 1e2)
pl.ylim(1e-5, 1e3)

pl.title("$m_\chi = 10\,\mathrm{GeV}$")
pl.legend(loc='best',fancybox = True, ncol=3)
pl.savefig(" plot 3")
pl.show()

m_x = 100 #DM mass in GeV

E_list = np.logspace(-3, 2,1000)
R_random = np.zeros((len(E_list), 10))

pl.figure(figsize=(7,5))

for i in range(10):
    cp_random = 1e-10*np.random.randn(20)
    cn_random = 1e-10*np.random.randn(20)
    cp_random[0] = 0
    cn_random[0] = 0
    
    R_random[:,i] = DMU.dRdE_NREFT(E_list, m_x, cp_random, cn_random, "Xe131") 
    pl.loglog(E_list, np.abs(R_random[:,i]))
    
pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')

pl.show()

#Spin-dependent coupling
cp_SD = np.zeros(20)
cp_SD[3] = 1.0

cn_SD = np.zeros(20)

m_x = 100 #DM mass in GeV

E_list = np.logspace(-3, 2,1000)

pl.figure()
pl.plot(E_list, 1e-3*DMU.dRdE_NREFT(E_list, m_x, cp_SD, cn_SD, "Xe131", j_x = 0.5),'r-' , lw=1.5,label ='Spin-1/2')
pl.plot(E_list, 1e-3*DMU.dRdE_NREFT(E_list, m_x, cp_SD, cn_SD, "Xe131", j_x = 1.0),'b-' , lw=1.5,label ='Spin-1')
pl.xlabel(r'$E_R$ [keV]')
#pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
#pl.title('Random NREFT interactions')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [arb. units]')

pl.legend(loc='best',fancybox = True)
pl.show()

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

#SI cross section
sig = 1e-46

pl.figure(figsize=(7,5))
pl.loglog(E_list, DMU.dRdE_standard(E_list, N_p_Xe, N_n_Xe, m_x, sig), lw=1.5, label='Spin-independent')
pl.loglog(E_list, DMU.dRdE_millicharge(E_list, m_x, eps, "Xe131"), lw=1.5, label='Millicharge DM')
pl.loglog(E_list, DMU.dRdE_magnetic(E_list, m_x, mu_x, "Xe131"), lw=1.5, label='Magnetic DM')
pl.loglog(E_list, DMU.dRdE_anapole(E_list, m_x, c_A, "Xe131"), lw=1.5, label='Anapole DM')

pl.xlabel(r'$E_R$ [keV]')
pl.ylabel(r'$\mathrm{d}R/\mathrm{d}E_R$ [keV$^{-1}$ kg$^{-1}$ day$^{-1}$]')
pl.legend(loc = 'best')
pl.show()