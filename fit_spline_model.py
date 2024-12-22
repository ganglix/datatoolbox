import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import t, f

def fit_spline_with_confidence(x, y, smoothing_factor=None, alpha=0.05):
    """
    Fit a spline model and calculate pointwise confidence intervals for the fitting curve.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    smoothing_factor (float or None): Smoothing factor for the spline. If None, an optimized default value will be used.
    alpha (float): Significance level (default is 0.05 for 95% confidence).

    Returns:
    tuple:
        - spline (UnivariateSpline): The fitted spline model.
        - CI_fit (numpy array): Pointwise confidence intervals for the fitting curve.
        - y_pred (numpy array): Predicted values from the spline.
        - r_squared (float): Coefficient of determination (R²) for the fit.
        - p_value (float): P-value indicating the significance of the fit.
    """
    # Fit the spline model with the specified smoothing factor or an optimized default
    spline = UnivariateSpline(x, y, s=smoothing_factor)

    # Predicted values
    y_pred = spline(x)

    # Residuals
    residuals = y - y_pred

    # Standard error of residuals
    stderr = np.std(residuals)

    # t-value for the given alpha
    dof = len(x) - spline.get_knots().size
    t_value = t.ppf(1 - alpha / 2, dof)

    # Compute pointwise confidence intervals for the fitting curve
    h = np.array([spline.derivatives(xi)[0] for xi in x])  # Derivative at each x (local slope)
    leverage = (1 / len(x)) + (h - np.mean(h))**2 / np.sum((h - np.mean(h))**2)

    CI_fit = t_value * stderr * np.sqrt(leverage)

    # Calculate R²
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calculate F-statistic and p-value
    num_params = len(spline.get_knots())
    f_stat = ((ss_total - ss_residual) / (num_params - 1)) / (ss_residual / dof)
    p_value = 1 - f.cdf(f_stat, num_params - 1, dof)

    return spline, CI_fit, y_pred, r_squared, p_value

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 3 * x + 7 + np.random.normal(0, 3, size=len(x))

    # Adjust smoothing factor to reduce overfitting
    optimal_smoothing_factor = 500  # Example of a less aggressive smoothing

    # Fit the spline and calculate confidence intervals
    spline, CI_fit, y_pred, r_squared, p_value = fit_spline_with_confidence(x, y, smoothing_factor=optimal_smoothing_factor)

    # Print confidence intervals, R², and p-value
    print(f"Pointwise Confidence Interval for Fitting Curve: {CI_fit}")
    print(f"R²: {r_squared:.4f}")
    print(f"p-value: {p_value:.4e}")

    # Plot the data, spline fit, and confidence bounds
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data", color="lightgrey", edgecolor="darkgrey", linewidth=1, alpha=1)
    plt.plot(x, y_pred, label="Spline Fit", color="black")
    plt.fill_between(x, y_pred - CI_fit, y_pred + CI_fit, linewidth=0, color="grey", alpha=0.3, label="95% CI (Fitting Curve)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Spline Regression with Pointwise Confidence Intervals")
    plt.legend()
    plt.show()
