import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calculate_confidence_interval(x, y, model, alpha=0.05):
    """
    Calculate the confidence interval for a linear model.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    model (LinearRegression): Fitted linear regression model.
    alpha (float): Significance level (default is 0.05 for 95% confidence).

    Returns:
    tuple: Predicted y values and confidence intervals.
    """
    from scipy.stats import t

    # Predictions
    y_pred = model.predict(x.reshape(-1, 1))

    # Residual standard error
    residuals = y - y_pred
    dof = len(x) - 2  # Degrees of freedom
    SE = np.sqrt(np.sum(residuals**2) / dof)

    # t-value for the given alpha
    t_value = t.ppf(1 - alpha / 2, dof)

    # Standard error of the mean predictions
    mean_x = np.mean(x)
    n = len(x)
    SE_mean = SE * np.sqrt(1 / n + (x - mean_x)**2 / np.sum((x - mean_x)**2))

    # Confidence interval
    CI = t_value * SE_mean

    return y_pred, CI, SE

# Main code
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 3 * x + 7 + np.random.normal(0, 3, size=len(x))

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    # Extract fitted parameters
    slope = model.coef_[0]
    intercept = model.intercept_

    # Calculate residual standard error
    y_pred, CI, SE = calculate_confidence_interval(x, y, model)

    # Variance of x
    mean_x = np.mean(x)
    sum_sq_x = np.sum((x - mean_x)**2)

    # Standard error for slope and intercept
    SE_slope = SE / np.sqrt(sum_sq_x)
    SE_intercept = SE * np.sqrt(np.sum(x**2) / (len(x) * sum_sq_x))

    # Print fitted parameters and their standard errors
    print(f"Slope: {slope:.4f}, SE: {SE_slope:.4f}")
    print(f"Intercept: {intercept:.4f}, SE: {SE_intercept:.4f}")

    # Plot the data, fit, and confidence bounds
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data", color="blue", alpha=0.7)
    plt.plot(x, y_pred, label="Fitted Line", color="red")
    plt.fill_between(x, y_pred - CI, y_pred + CI, color="gray", alpha=0.3, label="95% Confidence Interval")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression with 95% Confidence Interval")
    plt.legend()
    plt.show()
