import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# Constants
c = 3.0e8  # m/s
hbar_c = 197.3e-15  # GeV*m
GeV_to_kg = 1.7827e-27  # GeV/c^2 to kg
rho_0 = 0.3 * GeV_to_kg / (1e-2)**3  # GeV/cm^3 to kg/m^3
v0 = 220e3    # m/s
v_E = 232e3   # m/s
v_esc = 544e3 # m/s
N0 = 6.022e26  # 1/mol
sqrt_pi = np.sqrt(np.pi)

# Detector nucleus
A = 1
m_A_GeV = 1.0
m_A = m_A_GeV * GeV_to_kg  # kg

# WIMP masses and cross section
dict_masses = [0.1, 0.5, 1.0, 5.0, 10.0]  # GeV
sigma_n = 1e-36  # cm^2

# Reduced mass
def mu_kg(m1, m2):
    return m1 * m2 / (m1 + m2)

# v_min in m/s for given E_R in keV
def v_min(E_R_keV, m_x_kg):
    E_J = E_R_keV * 1e3 * 1.602e-19
    mu = mu_kg(m_x_kg, m_A)
    return np.sqrt(m_A * E_J / (2 * mu**2))

# Helm form factor squared
def F2(E_R_keV):
    E_R_J = E_R_keV * 1e3 * 1.602e-19
    q = np.sqrt(2 * m_A * E_R_J) / hbar_c
    A13 = A**(1/3)
    s = 0.9e-15  # m
    r = 1.2 * A13 * 1e-15
    r_n_sq = r**2 - 5 * s**2
    r_n_sq = np.maximum(r_n_sq, 0.0)  # ensure non-negative for sqrt
    r_n = np.sqrt(r_n_sq)
    qr_n = q * r_n
    qs = q * s
    with np.errstate(divide='ignore', invalid='ignore'):
        F_q = 3 * (np.sin(qr_n) - qr_n * np.cos(qr_n)) / (qr_n**3)
        F_q *= np.exp(-0.5 * qs**2)
    F_q = np.nan_to_num(F_q, nan=1.0, posinf=1.0, neginf=1.0)
    return F_q**2

# Velocity integral η(v_min)
def eta(vmin):
    x_min = vmin / v0
    x_E = v_E / v0
    x_esc = v_esc / v0
    N = erf(x_esc) - 2 * x_esc / sqrt_pi * np.exp(-x_esc**2)
    term = ((erf(x_min + x_E) - erf(x_min - x_E)) / (2 * x_E))-np.exp(-x_esc**2)
    term = np.maximum(term, 0.0)  # avoid negative or small values
    return np.where(vmin < v_esc, term / N, 0)

# Final differential rate with all corrections
def dR_dE(E_R_keV, m_x_kg):
    mu_xA = mu_kg(m_x_kg, m_A)
    mu_xn = mu_kg(m_x_kg, GeV_to_kg)  # neutron ~1 GeV

    # sigma_chiA from Eq. (11)
    sigma_chiA = sigma_n * 1e-4 * (mu_xA / mu_xn)**2 * A**2  # cm^2 to m^2

    # Final corrected coefficient
    coeff = ((1 / sqrt_pi) * (N0 / A) * (rho_0 * m_A * sigma_chiA)) / (m_x_kg * mu_xA**2 * v0)

    return coeff * F2(E_R_keV) * eta(v_min(E_R_keV, m_x_kg)) * 86400  # /day

# Plotting
E_R = np.logspace(-3, 1, 500)

plt.figure()
for m_x_GeV in dict_masses:
    m_x_kg = m_x_GeV * GeV_to_kg
    plt.plot(E_R, dR_dE(E_R, m_x_kg), label=f'{m_x_GeV} GeV')

plt.xlabel('Recoil Energy $E_R$ (keV)')
plt.ylabel(r'$dR/dE_R$ (keV$^{-1}$ day$^{-1}$ kg$^{-1}$)')
plt.xscale('log')
plt.yscale('log')
plt.legend(title='WIMP Mass')
plt.title(r'$\frac{dR}{dE_R}$ with $v_0/E_0$ and full prefactor')
plt.tight_layout()
plt.show()

# --- Check values of individual functions ---
print("\n--- Sample Outputs for Debugging ---")
sample_E_R = 1.0  # keV
sample_m_x_GeV = 1.0
sample_m_x_kg = sample_m_x_GeV * GeV_to_kg

mu_val = mu_kg(sample_m_x_kg, m_A)
vmin_val = v_min(sample_E_R, sample_m_x_kg)
F2_val = F2(sample_E_R)
eta_val = eta(vmin_val)
mu_xn = mu_kg(sample_m_x_kg, GeV_to_kg)
sigma_chiA = sigma_n * 1e-4 * (mu_val / mu_xn)**2 * A**2
coeff_val =( (1 / sqrt_pi) * (N0 / A) * (rho_0 * m_A* sigma_chiA)) / (sample_m_x_kg * mu_val**2 * v0)
dR_dE_val = coeff_val * F2_val * eta_val * 86400

print(f"mu_kg = {mu_val:.3e} kg")
print(f"v_min = {vmin_val:.3e} m/s")
print(f"F2(E_R) = {F2_val:.3e}")
print(f"eta(v_min) = {eta_val:.3e}")
print(f"coeff = {coeff_val:.3e}")
print(f"dR/dE = {dR_dE_val:.3e} events/kg/day/keV")
