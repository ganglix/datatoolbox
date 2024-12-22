# DATATOOLBOX

DATATOOLBOX is a Python package designed for data modeling, analysis, and visualization. It includes functionalities for polynomial fitting, confidence interval computation, spline fitting, and more. This toolbox is useful for researchers and engineers working with data-driven models.

## Features
- **Polynomial Fitting**: Fit polynomial models to data and calculate confidence intervals for the predictions.
- **Spline Fitting**: Fit splines to data and calculate confidence intervals for the fitted curve.
- **Visualization**: Plot fitted models alongside confidence intervals and the original data.
- **Customizability**: Easily extend the provided models for specific use cases.

## Installation
To install the toolbox locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ganglix/DATATOOLBOX.git
   ```

2. Navigate to the repository directory:
   ```bash
   cd DATATOOLBOX
   ```

3. Install the package locally in editable mode:
   ```bash
   pip install -e .
   ```

## Usage
### Example: Polynomial Fitting and Confidence Intervals
The following code demonstrates how to fit a polynomial model to data and calculate confidence intervals for the fitted curve.

```python
from fit_general_poly_model import polynomial_model, fit_and_calculate_intervals, 

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
```

### Example: Spline Fitting
The toolbox also supports spline fitting. Refer to the `fit_spline_model.py` for an example implementation.

## Repository Structure
```
DATATOOLBOX/
├── fit_general_poly_model.py        # Polynomial fitting and CI calculation
├── fit_linear_sklearn_model.py      # Linear regression using sklearn
├── fit_spline_model.py              # Spline fitting and confidence interval calculation
├── test_fit_general_poly_model.py   # Unit tests for polynomial fitting
├── README.md                        # Documentation
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any inquiries or support, contact [ganglix@gmail.com].

