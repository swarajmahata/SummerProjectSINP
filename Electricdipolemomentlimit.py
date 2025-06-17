import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import WIMpy.DMUtils as DMU
from interpolator import Interpolator, CubicInterpolator
import csv

# --- Nuclear mass numbers ---
Avals = {"C12": 12, "F19": 19}

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

xF, yF = load_csv_unique("Bubble Nucleation Efficiency_Flurin.csv")
xC, yC = load_csv_unique("Bubble Nucleation Efficiency_Carbon.csv")
eff_F = make_eff_func(xF, yF)
eff_C = make_eff_func(xC, yC)

# --- Constants ---
alpha = 0.007297
e = np.sqrt(4 * np.pi * alpha)
d0 = 2.8e-8
threshold = 2.45     # keV
exposure = 1404      # kg-day
mass_f = 0.8084 * 52
mass_c = 0.1916 * 52

# --- Electric dipole recoil rate ---
def dRdE_electric(E, m_x, d_x, target):
    A = Avals[target]
    amu = 931.5e3
    q = np.sqrt(2 * A * amu * E)
    q2 = (q * 1e-6)**2

    cp = [E * 0.0 for _ in range(20)]
    cn = [E * 0.0 for _ in range(20)]

    cp[10] = 2.0 * e * d_x / q2
    cn[10] = 2.0 * e * d_x / q2

    return DMU.dRdE_NREFT(E, m_x, cp, cn, target)

# --- Compute σ vs m_x ---
m_x_vals = np.logspace(0, 3, 300)
sigma_vals = []

for m_x in m_x_vals:
    try:
        def integrand(E):
            effF = eff_F(E / threshold)
            effC = eff_C(E / threshold)
            dRdE_f = dRdE_electric(E, m_x, d0, "F19") * effF
            dRdE_c = dRdE_electric(E, m_x, d0, "C12") * effC
            return mass_f * dRdE_f + mass_c * dRdE_c

        R_exp, _ = quad(integrand, threshold, 200.0, epsabs=1e-6, epsrel=1e-3)
        sigma = 2.3 / (R_exp * exposure) if R_exp > 1e-10 else np.nan
        sigma_vals.append(sigma)
        print(f"m_x: {m_x:.2f} GeV, R_exp: {R_exp:.4e}, σ: {sigma:.4e} pb")
    except Exception as e:
        print(f"Error at m_x = {m_x:.2f} GeV: {e}")
        sigma_vals.append(np.nan)

# --- Plot σ vs m_x ---
plt.figure(figsize=(7, 5))
plt.loglog(m_x_vals, sigma_vals, color='magenta', linewidth=2, label="C$_3$F$_8$ Electric")
plt.xlabel("DM Mass $m_\\chi$ [GeV]")
plt.ylabel(r"$\sigma^{\mathrm{SI}}_{\chi n,90}$ [pb]")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.title("Electric Dipole Cross-section Limit", loc='left')
plt.legend()
plt.tight_layout()
plt.savefig("Sigma_vs_Mass_ElectricDipole.png")
plt.show()
