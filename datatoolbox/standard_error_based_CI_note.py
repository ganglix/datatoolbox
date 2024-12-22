# Using only SE (standard error of fitted parameters) ignores correlations between parameters (off-diagonal elements of the covariance matrix).
# This leads to overestimation of CI width, especially in complex models where parameters are interdependent.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.optimize import curve_fit

def polynomial_model(x, *params):
    return sum(p * x**i for i, p in enumerate(params))

def fit_and_calculate_intervals(x, y, model, p0, alpha=0.05):
    from scipy.stats import t

    popt, pcov = curve_fit(model, x, y, p0=p0)
    param_errors = np.sqrt(np.diag(pcov))

    dof = len(x) - len(popt)
    t_value = t.ppf(1 - alpha / 2, dof)

    y_pred = model(x, *popt)

    jacobian = np.vstack([x**i for i in range(len(popt))]).T
    SE_mean = np.sqrt(np.sum((jacobian @ pcov) * jacobian, axis=1))
    CI_fit = t_value * SE_mean

    return popt, param_errors, CI_fit, y_pred, pcov

def calculate_ci_band_from_errors(x, popt, param_errors, model, alpha=0.05):
    """
    Reproduce the confidence interval band using standard errors of parameters only.

    Parameters:
    x (numpy array): Independent variable.
    popt (list): Optimized parameters from the fit.
    param_errors (list): Standard errors of the parameters.
    model (function): Polynomial model function.
    alpha (float): Confidence level (default 0.05).

    Returns:
    numpy array: Confidence interval band.
    """
    param_errors = np.array(param_errors)  # Ensure param_errors is a numpy array
    dof = len(x) - len(popt)
    t_value = t.ppf(1 - alpha / 2, dof)

    # Calculate the Jacobian matrix
    jacobian = np.vstack([x**i for i in range(len(popt))]).T

    # Variance propagation using parameter standard errors (diagonal only)
    param_variance = param_errors**2
    variance_y = np.sum((jacobian**2) * param_variance, axis=1)

    # Calculate confidence interval band
    ci_band = t_value * np.sqrt(variance_y)
    return ci_band

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 3 * x + 7 + np.random.normal(0, 3, size=len(x))

# Fit the model and calculate confidence intervals
popt, param_errors, CI_fit_original, y_pred, pcov = fit_and_calculate_intervals(x, y, polynomial_model, p0=[1, 1, 1])

# Recalculate confidence interval band using parameter errors
ci_band_from_errors = calculate_ci_band_from_errors(x, popt, param_errors, polynomial_model)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(x, y_pred, label="Fitted Polynomial", color="blue")
plt.fill_between(
    x, y_pred - CI_fit_original, y_pred + CI_fit_original,
    color="lightblue", alpha=0.5, label="CI (Original)"
)
plt.fill_between(
    x, y_pred - ci_band_from_errors, y_pred + ci_band_from_errors,
    color="gray", alpha=0.3, label="CI (Reproduced from Errors)"
)
plt.scatter(x, y, label="Data", color="black", s=10)
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Comparison of Original and Reproduced Confidence Intervals")
plt.show()
