import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def polynomial_model(x, *params):
    """
    Define a polynomial model: y = a0 + a1*x + a2*x^2 + ...

    Parameters:
    x (numpy array): Independent variable.
    *params: Coefficients of the polynomial.

    Returns:
    numpy array: Dependent variable values.
    """
    return sum(p * x**i for i, p in enumerate(params))

def fit_and_calculate_intervals(x, y, model, p0, alpha=0.05):
    """
    Fit a polynomial model and calculate confidence intervals, R², and p-value.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    model (function): Model function.
    p0 (list): Initial guess for the parameters.
    alpha (float): Significance level (default is 0.05 for 95% confidence).

    Returns:
    tuple: Confidence intervals, optimized parameters, parameter standard errors, R², and p-value.
    """
    from scipy.stats import t, f

    # Fit the model using curve_fit
    popt, pcov = curve_fit(model, x, y, p0=p0)

    # Extract standard errors of the parameters
    param_errors = np.sqrt(np.diag(pcov))

    # Degrees of freedom
    dof = len(x) - len(popt)

    # t-value for the given alpha
    t_value = t.ppf(1 - alpha / 2, dof)

    # Predictions
    y_pred = model(x, *popt)

    # Calculate R²
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calculate F-statistic and p-value
    num_params = len(popt)
    f_stat = ((ss_total - ss_residual) / (num_params - 1)) / (ss_residual / dof)
    p_value = 1 - f.cdf(f_stat, num_params - 1, dof)

    # Calculate the standard error of the mean predictions
    jacobian = np.vstack([x**i for i in range(len(popt))]).T
    SE_mean = np.sqrt(np.sum((jacobian @ pcov) * jacobian, axis=1))

    # Confidence interval
    CI_fit = t_value * SE_mean

    return popt, param_errors, CI_fit, r_squared, p_value

# Main code, use example
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 3 * x + 7 + np.random.normal(0, 3, size=len(x))

    # Fit the model and calculate confidence intervals, R², and p-value
    popt, param_errors, CI_fit, r_squared, p_value = fit_and_calculate_intervals(x, y, polynomial_model, p0=[1, 1, 1])
    y_pred = polynomial_model(x, *popt)

    # Print fitted parameters, their standard errors, R², and p-value
    print("Fitted Parameters and Standard Errors:")
    for i, (param, error) in enumerate(zip(popt, param_errors)):
        print(f"Coefficient a{i}: {param:.4f}, SE: {error:.4f}")
    print(f"R²: {r_squared:.4f}")
    print(f"p-value: {p_value:.4e}")

    # Plot the data, fit, and confidence bounds
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data", color="lightgrey", edgecolor="darkgrey", linewidth=1, alpha=1)
    plt.plot(x, y_pred, label="Fitted Polynomial", color="black")
    plt.fill_between(x, y_pred - CI_fit, y_pred + CI_fit, linewidth=0, color="gray", alpha=0.3, label="95% Confidence Interval (fit)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polynomial Regression with 95% Confidence Interval")
    plt.legend()
    plt.show()



# test the code
# Fitted Parameters and Standard Errors:
# Coefficient a0: 8.2423, SE: 1.1270
# Coefficient a1: 2.1836, SE: 0.5212
# Coefficient a2: 0.0642, SE: 0.0504
# R²: 0.9064
# p-value: 1.1102e-16


