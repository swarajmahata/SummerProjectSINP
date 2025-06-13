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

# --- Create interpolator ---
def make_eff_func(x, y):
    method = CubicInterpolator()
    interpolator = Interpolator(method)
    interp = interpolator.interpolate(x, y)
    return lambda E_ratio: np.clip(interp(E_ratio), 1e-3, 1.0)

# --- Efficiency functions ---
xF, yF = load_csv_unique("Bubble Nucleation Efficiency_Flurin.csv")
xC, yC = load_csv_unique("Bubble Nucleation Efficiency_Carbon.csv")
eff_F = make_eff_func(xF, yF)
eff_C = make_eff_func(xC, yC)

# --- Parameters ---
threshold = 2.45     # keV
exposure = 1404     # kg-day
eps0 = 2.2e-8        # reference millicharge (e units)

# --- Mass fractions ---
mass_f = 0.8084 * 52  # F19
mass_c = 0.1916 * 52  # C12

# --- DM mass range ---
m_x_vals = np.logspace(0, 3, 300)
sigma_vals = []

# --- Compute σ for each m_x ---
for m_x in m_x_vals:
    try:
        def integrand(E):
            effF = eff_F(E / threshold)
            effC = eff_C(E / threshold)
            dRdE_f = DMU.dRdE_millicharge(E, m_x, eps0, "F19") * effF
            dRdE_c = DMU.dRdE_millicharge(E, m_x, eps0, "C12") * effC
            return mass_f * dRdE_f + mass_c * dRdE_c

        R_exp, _ = quad(integrand, threshold, 200.0, epsabs=1e-6, epsrel=1e-3)
        sigma = 2.3 / (R_exp * exposure) if R_exp > 1e-10 else np.nan
        sigma_vals.append(sigma)

        print(f"m_x: {m_x:.2f} GeV, R_exp: {R_exp:.4e}, sigma: {sigma:.4e} pb")

    except Exception as e:
        print(f"Error at m_x = {m_x:.2f} GeV: {e}")
        sigma_vals.append(np.nan)

# --- Convert σ to ε (coupling) limits ---
eps_vals = [10 * eps0 * np.sqrt(s) if s is not np.nan else np.nan for s in sigma_vals]

# --- Plot σ vs m_x ---
plt.figure(figsize=(7, 5))
plt.loglog(m_x_vals, sigma_vals, color='brown', linewidth=2, label="PICO-60 C$_3$F$_8$")
plt.xlabel("DM Mass $m_\\chi$ [GeV]")
plt.ylabel(r"$\sigma^{\mathrm{SI}}_{\chi n,90}$ [pb]")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.title("Millicharge Cross-section Limit", loc='left')
plt.legend()
plt.tight_layout()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray')
plt.tick_params(axis='both', which='major', length=7)
plt.savefig("Sigma_vs_Mass_Millicharge_EffCorrected.png")
plt.show()

# --- Plot ε vs m_x ---
plt.figure(figsize=(7, 5))
plt.loglog(m_x_vals, eps_vals, color='darkred', linewidth=2, label=r"PICO-60 C$_3$F$_8$")
plt.xlabel(r"DM Mass $m_\chi$ [GeV]")
plt.ylabel(r"Millicharge Coupling $\varepsilon$")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.title("Millicharge Coupling Limit", loc='left')
plt.legend()
plt.tight_layout()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray')
plt.tick_params(axis='both', which='major', length=7)
plt.savefig("Epsilon_vs_Mass_Millicharge_EffCorrected.png")
plt.show()
