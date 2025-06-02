import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf, spherical_jn
import csv
from interpolator import Interpolator, CubicInterpolator

# --- Constants ---
sigma_n = 1e-36
c = 3.0e8
GeV_to_keV = 1e6 / c**2
N0 = 6.023e26
fm_to_m = 1e-15
rho_0 = (0.3 * (1e6 / c**2)) / (1e-2)**3
v0 = 220e3
v_E = 232e3
v_esc = 544e3
E_R_th = 3.84  # keV threshold

# --- CSV loader ---
def load_csv_unique(filename):
    data = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                x_val, y_val = float(row[0]), float(row[1])
                if x_val not in data:
                    data[x_val] = y_val
            except:
                continue
    x_unique, y_unique = zip(*sorted(data.items()))
    return list(x_unique), list(y_unique)

# --- Interpolated efficiency ---
def make_eff_func(x_eff, y_eff):
    interp = Interpolator(CubicInterpolator()).interpolate(x_eff, y_eff)
    return lambda E_ratio: np.clip(interp(E_ratio), 1e-3, 1.0)

# --- Load efficiency curves ---
x_eff_C, y_eff_C = load_csv_unique("Bubble Nucleation Efficiency_Carbon.csv")
x_eff_F, y_eff_F = load_csv_unique("Bubble Nucleation Efficiency_Flurin.csv")

eta_C = make_eff_func(x_eff_C, y_eff_C)
eta_F = make_eff_func(x_eff_F, y_eff_F)

# --- Physics functions ---
def mu_kg(m1_kg, m2_kg):
    return m1_kg * m2_kg / (m1_kg + m2_kg)

def v_min(E_R_keV, m_x_kg, m_A):
    return np.sqrt(m_A * E_R_keV / (2 * mu_kg(m_x_kg, m_A)**2))

def N():
    x_esc = v_esc / v0
    return erf(x_esc) - (2 * x_esc / np.sqrt(np.pi)) * np.exp(-x_esc**2)

def F2(E_R_keV, A):
    GeV_to_kg, keV_to_J = 1.78266192e-27, 1.60218e-16
    c_val, a, s = 1.23 * A**(1/3) * fm_to_m, 0.52 * fm_to_m, 0.9 * fm_to_m
    r_n = np.sqrt(c_val**2 + (7/3)*(np.pi**2)*a**2 - 5*s**2)
    m_A_val = A * GeV_to_kg
    E_R_J = E_R_keV * keV_to_J
    q = np.sqrt(2 * m_A_val * E_R_J)
    qrn, qs = q * r_n, q * s
    j1 = spherical_jn(1, qrn)
    j1_over_qrn = np.where(np.abs(qrn) < 1e-8, 1/3, j1 / qrn)
    return (3 * j1_over_qrn * np.exp(-0.5 * qs**2))**2

def eta(vmin):
    term = erf((vmin + v_E) / v0) - erf((vmin - v_E) / v0)
    return term / (2.25 * v_E / v0) / N()

# --- Compute rate ---
def compute_rate(m_x_GeV, A, m_A_GeV, eta_T_func, multiplier):
    m_x_kg = m_x_GeV * GeV_to_keV
    m_A = m_A_GeV * GeV_to_keV
    mu_xA = mu_kg(m_x_kg, m_A)
    mu_xn = mu_kg(m_x_kg, 1 * GeV_to_keV)
    sigma_A = sigma_n * 1e-4 * A**2 * (mu_xA / mu_xn)**2
    coeff = (1 / np.sqrt(np.pi)) * (N0 / A) * (rho_0 * m_A * sigma_A) / (m_x_kg * mu_xA**2 * v0)
    E_R_max = (2 * m_A * m_x_kg**2 * v_esc**2) / (m_A + m_x_kg)**2

    if E_R_max < E_R_th:
        return 0.0

    def integrand(E_R):
        E_ratio = E_R / E_R_th
        return multiplier * coeff * 86400 * eta_T_func(E_ratio) * F2(E_R, A) * eta(v_min(E_R, m_x_kg, m_A))

    result, _ = quad(integrand, E_R_th, E_R_max, limit=500, epsabs=1e-6, epsrel=1e-4)
    return result

# --- WIMP mass range ---
m_x_vals = np.linspace(2, 5.0, 150)

# --- Calculate event rates ---
rates_C = [compute_rate(m_x, 12, 12, eta_C, 0.235) for m_x in m_x_vals]
rates_F = [compute_rate(m_x, 19, 19, eta_F, 0.745) for m_x in m_x_vals]
total_rates = [c + f for c, f in zip(rates_C, rates_F)]

# --- Print results ---
print("Mass (GeV)\tR_C\t\tR_F\t\tTotal")
for m, rc, rf, rt in zip(m_x_vals, rates_C, rates_F, total_rates):
    print(f"{m:.2f}\t\t{rc:.4e}\t{rf:.4e}\t{rt:.4e}")

# --- Plot ---
plt.figure()
plt.plot(m_x_vals, rates_C, label="12C", linestyle='--', color='green')
plt.plot(m_x_vals, rates_F, label="19F", linestyle='-.', color='orange')
plt.plot(m_x_vals, total_rates, label="Sum", linestyle='-', color='red')
plt.yscale('log')
plt.xlabel("Mass (GeV)")
plt.ylabel("R (kg$^{-1}$ day$^{-1}$)")
plt.title("T = 55$^\circ$C, $\eta_T$ = 100%, $E_{th}$ = 3.84 keV")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("EventRate_T55_Eth3.84.png")
plt.show()
