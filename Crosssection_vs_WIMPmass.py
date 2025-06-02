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
E_R_th_default = 0.19  # keV
E_R_th_seitz = 1.92    # keV

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

# --- Make interpolated efficiency function ---
def make_eff_func(x_eff, y_eff):
    interp = Interpolator(CubicInterpolator()).interpolate(x_eff, y_eff)
    return lambda E_ratio: np.clip(interp(E_ratio), 1e-3, 1.0)

# --- Hydrogen step efficiency ---
def hydrogen_eff(E_ratio):
    return 1.0 if E_ratio >= 1.0 else 0.0

# --- Efficiency assignment ---
eff_functions = {
    "H": hydrogen_eff,
    "C": make_eff_func(*load_csv_unique("Bubble Nucleation Efficiency_Carbon.csv")),
    "F": make_eff_func(*load_csv_unique("Bubble Nucleation Efficiency_Flurin.csv"))
}

# --- Physics ---
def mu_kg(m1_kg, m2_kg): return m1_kg * m2_kg / (m1_kg + m2_kg)
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
def eta(vmin): return np.where(vmin < v_esc, ((erf((vmin + v_E)/v0) - erf((vmin - v_E)/v0)) / (2.25 * v_E / v0)) - np.exp(-(v_esc / v0)**2), 0) / N()

# --- Targets ---
targets = {
    "H": {"A": 1, "m_A_GeV": 1, "multiplier": 0.02, "style": ':'},
    "C": {"A": 12, "m_A_GeV": 12, "multiplier": 0.235, "style": (0, (1, 1))},
    "F": {"A": 19, "m_A_GeV": 19, "multiplier": 0.745, "style": '--'}
}

# --- Function to compute total rates ---
def compute_total_rates(E_R_th):
    rates = []
    for m_x_GeV in m_x_vals:
        m_x_kg = m_x_GeV * GeV_to_keV
        total_rate = 0
        for key, props in targets.items():
            A, m_A = props["A"], props["m_A_GeV"] * GeV_to_keV
            mu_xA = mu_kg(m_x_kg, m_A)
            mu_xn = mu_kg(m_x_kg, 1 * GeV_to_keV)
            sigma_A = sigma_n * 1e-4 * A**2 * (mu_xA / mu_xn)**2
            coeff = (1 / np.sqrt(np.pi) * (N0 / A) * (rho_0 * m_A * sigma_A)) / (m_x_kg * mu_xA**2 * v0)
            E_R_max = (2 * m_A * m_x_kg**2 * v_esc**2) / (m_A + m_x_kg)**2

            if E_R_max < E_R_th:
                rate = 0
            else:
                def integrand(E_R):
                    E_ratio = E_R / E_R_th
                    eff = eff_functions[key](E_ratio)
                    return eff * props["multiplier"] * coeff * 86400 * F2(E_R, A) * eta(v_min(E_R, m_x_kg, m_A))
                rate, _ = quad(integrand, E_R_th, E_R_max, limit=500, epsabs=1e-6, epsrel=1e-4)

            total_rate += rate
        rates.append(total_rate)
    return rates

# --- Mass range ---
m_x_vals = np.linspace(0.2, 5.0, 200)

# --- Compute for both thresholds ---
rates_default = compute_total_rates(E_R_th_default)
rates_seitz = compute_total_rates(E_R_th_seitz)

# --- Exposure ---
exposure = 1000  # kg.day
sigma_limits_default = [2.3 / (R * exposure) if R > 0 else np.nan for R in rates_default]
sigma_limits_seitz = [2.3 / (R * exposure) if R > 0 else np.nan for R in rates_seitz]

# --- Print Rexp values ---
for m_x, R in zip(m_x_vals, rates_default):
    print(f"Mass: {m_x:.2f} GeV, R_exp (0.19 keV): {R:.4e} kg^-1 day^-1")

# --- Plot ---
plt.figure()
plt.plot(m_x_vals, sigma_limits_default, linestyle='--', color='black', label=r"$E_{{R,\mathrm{{th}}}}=0.19$ keV")
plt.plot(m_x_vals, sigma_limits_seitz, linestyle='-', color='red', label=r"$E_{{R,\mathrm{{th}}}}=1.92$ keV (Seitz)")
plt.xlabel("Mass (GeV)")
plt.ylabel(r"$\sigma^{\mathrm{SI}}_{\chi,n,90}$ (pb)")
plt.yscale("log")
plt.xlim(0, 5)
plt.ylim(1e-9, 1e-1)
plt.grid(True)
plt.title(r"90% C.L. Poisson Upper Limit\n$T = 55^\circ$C, $\eta_T = 100\%$, Exposure = 1000 kg.day")
plt.legend()
plt.tight_layout()
plt.savefig("UpperLimit_vs_Mass_T55_eta100_Seitz.png")
plt.show()
