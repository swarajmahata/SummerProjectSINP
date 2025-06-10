import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import WIMpy.DMUtils as DMU  # Make sure WIMpy is installed

# --- Parameters ---
threshold = 2.45     # keV
exposure = 1404     # kg-day
eps = 2.2e-8          # millicharge (e units)

# --- Mass fractions for C₃F₈ ---
mass_f = 0.8084*52  # Fluorine fraction (F19)
mass_c = 0.1916*52  # Carbon fraction (C12)

# --- DM mass range (log scale) ---
m_x_vals = np.logspace(0, 3, 300)  # 1 GeV to 1000 GeV

# --- Compute σ vs m_x for full C₃F₈ ---
sigma_vals = []

for m_x in m_x_vals:
    try:
        def integrand(E):
            dRdE_f = DMU.dRdE_millicharge(E, m_x, eps, "F19")
            dRdE_c = DMU.dRdE_millicharge(E, m_x, eps, "C12")
            return mass_f * dRdE_f + mass_c * dRdE_c

        R_exp, _ = quad(integrand, threshold, 200.0, epsabs=1e-6, epsrel=1e-3)
        sigma = 2.3 / (R_exp * exposure) if R_exp > 1e-10 else np.nan
        sigma_vals.append(sigma)

        # --- Debug print ---
        print(f"m_x: {m_x:.2f} GeV, R_exp: {R_exp:.4e} events/kg/day, sigma: {sigma:.4e} pb")

    except Exception as e:
        print(f"Error at m_x = {m_x:.2f} GeV: {e}")
        sigma_vals.append(np.nan)

# --- Plot ---
plt.figure(figsize=(7, 5))
plt.loglog(m_x_vals, sigma_vals, color='brown', linewidth=2, label="PICO-60 C$_3$F$_8$")

plt.xlabel("DM Mass $m_\\chi$ [GeV]")
plt.ylabel(r"$\sigma^{\mathrm{SI}}_{\chi n,90}$ [pb]")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.title("Millicharge", loc='left')
plt.legend()
plt.tight_layout()


# --- Axis ticks ---
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=4, color='gray')
plt.tick_params(axis='both', which='major', length=7)

plt.savefig("Sigma_vs_Mass_Millicharge")
plt.show()


