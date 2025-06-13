import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import WIMpy.DMUtils as DMU  # Make sure WIMpy is installed

# --- Parameters ---
threshold = 2.45     # keV
exposure = 1404      # kg-day
eps0 = 2.2e-8        # reference millicharge for spectrum calculation (to scale from)

# --- Mass fractions for C₃F₈ ---
mass_f = 0.8084 * 52  # F19 fraction
mass_c = 0.1916 * 52  # C12 fraction

# --- Mass range ---
m_x_vals = np.logspace(0, 3, 300)  # 1 GeV to 1000 GeV

# --- Calculate sigma with fixed eps0, then scale to get eps limits ---
sigma_vals = []

for m_x in m_x_vals:
    try:
        def integrand(E):
            dRdE_f = DMU.dRdE_millicharge(E, m_x, eps0, "F19")
            dRdE_c = DMU.dRdE_millicharge(E, m_x, eps0, "C12")
            return mass_f * dRdE_f + mass_c * dRdE_c

        R_exp, _ = quad(integrand, threshold, 200.0, epsabs=1e-6, epsrel=1e-3)
        sigma = 2.3 / (R_exp * exposure) if R_exp > 1e-10 else np.nan
        sigma_vals.append(sigma)
        print(f"m_x: {m_x:.2f} GeV, R_exp: {R_exp:.4e}, sigma: {sigma:.4e}")
    except Exception as e:
        print(f"Error at m_x = {m_x:.2f} GeV: {e}")
        sigma_vals.append(np.nan)

# --- Convert sigma to eps limits ---
# If sigma ∝ eps², then eps ∝ sqrt(sigma)
eps_vals = [10*eps0 * np.sqrt(s) if s is not np.nan else np.nan for s in sigma_vals]

# --- Plot ε vs mχ ---
plt.figure(figsize=(7, 5))
plt.loglog(m_x_vals, eps_vals, color='darkred', linewidth=2, label=r"PICO-60 C$_3$F$_8$")

plt.xlabel(r"DM Mass $m_\chi$ [GeV]")
plt.ylabel(r"Millicharge Coupling $\varepsilon$")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.title("Millicharge Limit", loc='left')
plt.legend()
plt.tight_layout()

# Axis ticks
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray')
plt.tick_params(axis='both', which='major', length=7)

plt.savefig("Epsilon_vs_Mass_Millicharge.png")
plt.show()
