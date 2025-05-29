import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import sys

# Usage: python function.py
# Plots for fixed sigma_n = 1 pb and various WIMP masses
dict_masses = [0.1, 0.5, 1.0, 5.0, 10.0]  # GeV
sigma_n = 1e-36  # 1 pb in cm^2

# Physical constants
c = 3.0e8  # speed of light (m/s)
hbar_c = 197.3e-15  # GeV·m
GeV_to_keV = 1e6/c**2 # 1 GeV/c^2 to kev/c2
N0 = 6.023e26 

rho_0 = (0.3 * (1e6/c**2)) / (1e-2)**3  # convert GeV/cm3 to keV/m3
v0 = 220e3    # m/s
v_E = 232e3   # m/s
v_esc = 544e3 # m/s

# Detector: hydrogen
A = 1
m_A_GeV = 1.0
m_A = m_A_GeV * GeV_to_keV

# Reduced mass in kg
def mu_kg(m1_kg, m2_kg):
    return m1_kg * m2_kg / (m1_kg + m2_kg)

# Minimal velocity (m/s) for recoil energy E_R (keV)
def v_min(E_R_keV, m_x_kg):
    E_J = E_R_keV 
    mu = mu_kg(m_x_kg, m_A)
    return np.sqrt(m_A * E_J / (2 * mu**2))

# Simple form factor for hydrogen (F=1)
def F2(_):
    return 3.0

# Velocity integral η(vmin)
def eta(vmin):
    x_min = vmin / v0
    x_E = v_E / v0
    x_esc = v_esc / v0
    N = erf(x_esc) - 2 * x_esc / np.sqrt(np.pi) * np.exp(-x_esc**2)
    term = (erf(x_min + x_E) - erf(x_min - x_E)) / (2 * x_E)
    return np.where(vmin < v_esc, term / N, 0)

# Differential rate dR/dE_R
# units: events/kg/day/keV
def dR_dE(E_R_keV, m_x_kg):
    mu_xA = mu_kg(m_x_kg, m_A)
    mu_xn = mu_kg(m_x_kg, 1 * GeV_to_keV)
    sigma_A = sigma_n * 1e-4 * A**2 * (mu_xA / mu_xn)**2  # cm2->m2
    coeff = (1 / np.sqrt(np.pi) * (N0 / A) * (rho_0 * m_A * sigma_A)) / (m_x_kg * mu_xA**2 * v0)  # /s units
    # convert to per day: *86400
    return coeff * 86400 * F2(E_R_keV) * eta(v_min(E_R_keV, m_x_kg))

# Plot
E_R = np.logspace(-3, 1, 500)

plt.figure()
for m_x_GeV in dict_masses:
    m_x_kg = m_x_GeV * GeV_to_keV
    plt.plot(E_R, dR_dE(E_R, m_x_kg), label=f'{m_x_GeV} GeV')

plt.xlabel('Recoil Energy $E_R$ (keV)')
plt.ylabel(r'$dR/dE_R$ (keV$^{-1}$ day$^{-1}$ kg$^{-1}$)')
plt.xscale('log')
plt.yscale('log')
plt.title(r'$\sigma_{\chi n}=1\ \mathrm{pb}$')
plt.legend(title='Mass')
plt.tight_layout()
plt.show()
