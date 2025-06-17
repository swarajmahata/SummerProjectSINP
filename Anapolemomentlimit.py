import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import WIMpy.DMUtils as DMU
from interpolator import Interpolator, CubicInterpolator
import csv

# --- Load CSV ---
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
    x, y = zip(*sorted(data.items()))
    return list(x), list(y)

# --- Interpolator ---
def make_eff_func(x, y):
    method = CubicInterpolator()
    interpolator = Interpolator(method)
    interp = interpolator.interpolate(x, y)
    return lambda E_ratio: np.clip(interp(E_ratio), 1e-3, 1.0)

# --- Load efficiency data ---
xF, yF = load_csv_unique("Bubble Nucleation Efficiency_Flurin.csv")
xC, yC = load_csv_unique("Bubble Nucleation Efficiency_Carbon.csv")
eff_F = make_eff_func(xF, yF)
eff_C = make_eff_func(xC, yC)

# --- Constants ---
alpha = 0.007297
e = np.sqrt(4 * np.pi * alpha)
gp = 5.59
gn = -3.83
g_a0 = 3.6e-8
threshold = 2.45     # keV
exposure = 1404      # kg-day
mass_f = 0.8084 * 52
mass_c = 0.1916 * 52

# --- Recoil rate ---
def dRdE_anapole_spin_corrected(E, m_x, c_A, target):
    cp = np.zeros(20)
    cn = np.zeros(20)
    cp[7] = -2.0 * e * c_A
    if target == "F19":
        cp[8] = -gp * c_A
    elif target in ["Xe129", "Xe131"]:
        cn[8] = -gn * c_A
    return DMU.dRdE_NREFT(E, m_x, cp, cn, target)

# --- Compute cross-section σ ---
m_x_vals = np.logspace(0, 3, 300)
sigma_vals = []

for m_x in m_x_vals:
    try:
        def integrand(E):
            effF = eff_F(E / threshold)
            effC = eff_C(E / threshold)
            dRdE_f = dRdE_anapole_spin_corrected(E, m_x, g_a0, "F19") * effF
            dRdE_c = dRdE_anapole_spin_corrected(E, m_x, g_a0, "C12") * effC
            return mass_f * dRdE_f + mass_c * dRdE_c

        R_exp, _ = quad(integrand, threshold, 200.0, epsabs=1e-6, epsrel=1e-3)
        sigma = 2.3 / (R_exp * exposure) if R_exp > 1e-10 else np.nan
        sigma_vals.append(sigma)
        print(f"m_x: {m_x:.2f} GeV, R_exp: {R_exp:.4e}, sigma: {sigma:.4e} pb")
    except Exception as e:
        print(f"Error at m_x = {m_x:.2f} GeV: {e}")
        sigma_vals.append(np.nan)

# --- Plot cross-section ---
plt.figure(figsize=(7, 5))
plt.loglog(m_x_vals, sigma_vals, color='purple', linewidth=2, label="C$_3$F$_8$ Anapole")
plt.xlabel("DM Mass $m_\\chi$ [GeV]")
plt.ylabel(r"$\sigma^{\mathrm{SI}}_{\chi n,90}$ [pb]")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.title("Anapole Cross-section Limit", loc='left')
plt.legend()
plt.tight_layout()
plt.savefig("Sigma_vs_Mass_Anapole.png")
plt.show()
