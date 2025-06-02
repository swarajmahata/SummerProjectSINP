import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf, spherical_jn
import csv
from scipy.interpolate import interp1d
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
E_R_th = 0.38  # keV

# --- Load Efficiency CSV ---
def load_csv_unique(filename):
    data = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                try:
                    x_val = float(row[0])
                    y_val = float(row[1])
                    if x_val not in data:
                        data[x_val] = y_val
                except ValueError:
                    continue
    x_unique, y_unique = zip(*sorted(data.items()))
    return list(x_unique), list(y_unique)

# --- Make efficiency function for each target ---
def make_eff_func(x_eff, y_eff, label):
    method = CubicInterpolator()
    interpolator = Interpolator(method)
    interp = interpolator.interpolate(x_eff, y_eff)
    return lambda E_ratio: np.clip(interp(E_ratio), 1e-3, 1.0)

# --- Target-specific files ---
efficiency_files = {
    "C": "Bubble Nucleation Efficiency_Carbon.csv",
    "F": "Bubble Nucleation Efficiency_Flurin.csv"
}

eff_functions = {}
for key, file in efficiency_files.items():
    x_eff, y_eff = load_csv_unique(file)
    eff_functions[key] = make_eff_func(x_eff, y_eff, key)

# --- Physics ---
def mu_kg(m1_kg, m2_kg):
    return m1_kg * m2_kg / (m1_kg + m2_kg)

def v_min(E_R_keV, m_x_kg, m_A):
    E_J = E_R_keV
    mu = mu_kg(m_x_kg, m_A)
    return np.sqrt(m_A * E_J / (2 * mu**2))

def N():
    x_esc = v_esc / v0
    return erf(x_esc) - (2 * x_esc / np.sqrt(np.pi)) * np.exp(-x_esc**2)

def F2(E_R_keV, A):
    GeV_to_kg = 1.78266192e-27
    keV_to_J = 1.60218e-16
    c_val = 1.23 * A**(1/3) * fm_to_m
    a = 0.52 * fm_to_m
    s = 0.9 * fm_to_m
    r_n = np.sqrt(c_val**2 + (7/3)*(np.pi**2)*a**2 - 5*s**2)
    m_A_val = A * GeV_to_kg
    E_R_J = E_R_keV * keV_to_J
    q = np.sqrt(2 * m_A_val * E_R_J)
    qrn = q * r_n
    qs = q * s
    j1 = spherical_jn(1, qrn)
    j1_over_qrn = np.where(np.abs(qrn) < 1e-8, 1/3, j1 / qrn)
    exp_term = np.exp(-0.5 * qs**2)
    return (3 * j1_over_qrn * exp_term) ** 2

def term(vmin):
    x_min = vmin / v0
    x_E = v_E / v0
    x_esc = v_esc / v0
    return ((erf(x_min + x_E) - erf(x_min - x_E)) / (2.25 * x_E)) - np.exp(-x_esc**2)

def eta(vmin):
    return np.where(vmin < v_esc, term(vmin) / N(), 0)

# --- Targets ---
targets = {
    "C": {"A": 12, "m_A_GeV": 12, "multiplier": 0.235, "style": 'dotted'},
    "F": {"A": 19, "m_A_GeV": 19, "multiplier": 0.745, "style": 'dashed'}
}

m_x_vals = np.linspace(0.8, 2.0, 100)
rates = {key: [] for key in targets}
rates["Total"] = []

# --- Integration ---
for m_x_GeV in m_x_vals:
    m_x_kg = m_x_GeV * GeV_to_keV
    total_rate = 0
    for key, props in targets.items():
        A = props["A"]
        m_A = props["m_A_GeV"] * GeV_to_keV
        multiplier = props["multiplier"]

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
                efficiency = eff_functions[key](E_ratio)
                print(f"Target: {key}, E_R: {E_R:.3f} keV, E_ratio: {E_ratio:.3f}, Efficiency: {efficiency:.4f}")
               
                return efficiency * multiplier * coeff * 86400 * F2(E_R, A) * eta(v_min(E_R, m_x_kg, m_A))
            rate, _ = quad(integrand, E_R_th, E_R_max, limit=200)

        rates[key].append(rate)
        total_rate += rate
    rates["Total"].append(total_rate)

# --- Plot ---
plt.figure()
for key, props in targets.items():
    plt.plot(m_x_vals, rates[key], linestyle=props["style"], linewidth=2,
             label=fr"$^{{{key}}}$: $E_{{R,\mathrm{{th}}}}={E_R_th}$ keV")

plt.plot(m_x_vals, rates["Total"], linestyle='solid', linewidth=2, label="Total rate")
plt.xlabel("Mass (GeV)")
plt.ylabel("Rate (kg$^{-1}$day$^{-1}$)")
plt.yscale("log")
plt.xlim(0.8, 2.0)
plt.ylim(0.1, 1e4)
plt.grid(True)
plt.title("eta=50 T=55")
plt.legend()
plt.tight_layout()

# --- Axis ticks ---
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray')
plt.tick_params(axis='both', which='major', length=7)

plt.savefig("Event Rate.png")
plt.show()


