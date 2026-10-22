import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Input Experimental Data
angles_deg = np.array([2.17, 4.46, 5.60, 6.16])
angles_rad = np.radians(angles_deg)
x_sin = np.sin(angles_rad)  # sin(theta) values

y_acc = np.array([0.354, 0.774, 0.992, 1.09])  # Measured accelerations (m/s^2)
y_err = np.array([0.0030, 0.0039, 0.0041, 0.0043])  # Experimental uncertainties (m/s^2)

# 2. Linear Regression with Intercept (y = m*x + c)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_sin, y_acc)

# Calculate uncertainty for intercept
n = len(x_sin)
df = n - 2
x_mean = np.mean(x_sin)
sum_sq_x = np.sum((x_sin - x_mean)**2)
residuals = y_acc - (slope * x_sin + intercept)
s_err = np.sqrt(np.sum(residuals**2) / df)
intercept_err = s_err * np.sqrt(1/n + (x_mean**2)/sum_sq_x)

# 3. Linear Regression Forced through Origin (y = g_forced * x)
g_forced = np.sum(x_sin * y_acc) / np.sum(x_sin**2)
# Propagate uncertainty of g_forced
var_g_forced = np.sum((x_sin**2) * (y_err**2)) / (np.sum(x_sin**2))**2
dg_forced = np.sqrt(var_g_forced)

# 4. Generate Plot Legend
plt.figure(figsize=(7.5, 5))
plt.errorbar(x_sin, y_acc, yerr=y_err, fmt='o', color='tab:red', label='Experimental Data', capsize=5, elinewidth=1.5, markeredgewidth=1.5)

# Plot fit lines
x_fit = np.linspace(0, 0.12, 100)
plt.plot(x_fit, slope * x_fit + intercept, color='tab:blue', linestyle='--',
         label=f'Fit with Intercept:\n  $g$ = {slope:.2f} \u00b1 {std_err:.2f} m/s\u00b2\n  $c$ = {intercept:.3f} \u00b1 {intercept_err:.3f} m/s\u00b2')
plt.plot(x_fit, g_forced * x_fit, color='tab:green', linestyle='-',
         label=f'Fit through Origin:\n  $g$ = {g_forced:.2f} \u00b1 {dg_forced:.2f} m/s\u00b2')

plt.xlabel('$\\sin(\\theta)$', fontweight='bold', fontsize=11)
plt.ylabel('Acceleration $a$ (m/s\u00b2)', fontweight='bold', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0, 0.12)
plt.ylim(0, 1.25)
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.show()
