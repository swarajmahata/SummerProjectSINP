# second edits

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import sys
from scipy.special import spherical_jn

# Usage: python function.py
# Plots for fixed sigma_n = 1 pb and various WIMP masses
dict_masses = [0.1, 0.5, 1.0, 5.0, 10.0]  # GeV
sigma_n = 1e-36  # 1 pb in cm^2

# Physical constants
c = 3.0e8  # speed of light (m/s)
GeV_to_keV = 1e6/c**2 # 1 GeV/c^2 to kev/c2
N0 = 6.023e26 
fm_to_m= 1e-15

rho_0 = (0.3 * (1e6/c**2)) / (1e-2)**3  # convert GeV/cm3 to keV/m3
v0 = 220e3    # m/s
v_E = 232e3   # m/s
v_esc = 544e3 # m/s

# Detector: hydrogen
A = float(sys.argv[1])
m_A_GeV = float(sys.argv[2])
particle = str(sys.argv[3])
m_A = m_A_GeV * GeV_to_keV

# Reduced mass in kg
def mu_kg(m1_kg, m2_kg):
    return m1_kg * m2_kg / (m1_kg + m2_kg)

# Minimal velocity (m/s) for recoil energy E_R (keV)
def v_min(E_R_keV, m_x_kg):
    E_J = E_R_keV 
    mu = mu_kg(m_x_kg, m_A)
    return np.sqrt(m_A * E_J / (2 * mu**2))



# Normalization factor N
def N():
    x_esc = v_esc / v0
    return erf(x_esc) - (2 * x_esc / np.sqrt(np.pi)) * np.exp(-x_esc**2)





def F2(E_R_keV, A=12):
    import numpy as np
    from scipy.special import spherical_jn

    GeV_to_kg = 1.78266192e-27
    keV_to_J = 1.60218e-16
    fm_to_m = 1e-15

    c = 1.23 * A**(1/3) * fm_to_m
    a = 0.52 * fm_to_m
    s = 0.9 * fm_to_m
    r_n = np.sqrt(c**2 + (7/3)*(np.pi**2)*a**2 - 5*s**2)

    m_A = A * GeV_to_kg
    E_R_J = E_R_keV * keV_to_J
    q = np.sqrt(2 * m_A * E_R_J)

    qrn = q * r_n
    qs = q * s

    # Vectorized safe j1_over_qrn
    j1 = spherical_jn(1, qrn)
    j1_over_qrn = np.where(np.abs(qrn) < 1e-8, 1/3, j1 / qrn)

    exp_term = np.exp(-0.5 * qs**2)
    F = 3 * j1_over_qrn * exp_term
    return F**2


    print(f"E_R_keV: {E_R_keV}, qrn: {qrn:.2e}, j1/q: {j1_over_qrn:.2e}, qs: {qs:.2e}, exp_term: {exp_term:.2e}, F2: {F2_val:.2e}")
    return F2_val


# Term in numerator of η(vmin) — using original form
def term(vmin):
    x_min = vmin / v0
    x_E = v_E / v0
    x_esc = v_esc / v0
    return ((erf(x_min + x_E) - erf(x_min - x_E)) / (2.25 * x_E)) - np.exp(-x_esc**2)

# Final η(vmin) expression as term/N
def eta(vmin):
    return np.where(vmin < v_esc, term(vmin) / N(), 0)

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
plt.title(r'$\sigma_{\chi n}=1\ \mathrm{pb}$. '+particle)
plt.legend(title='Mass')
plt.tight_layout()
plt.savefig(particle+"_spectrum.png")
plt.show()
