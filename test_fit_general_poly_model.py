import pytest
import sys
import os
import numpy as np

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fit_general_poly_model import polynomial_model, fit_and_calculate_intervals

@pytest.fixture
def synthetic_data():
    """Generate synthetic test data."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 3 * x + 7 + np.random.normal(0, 3, size=len(x))
    return x, y

def test_fit_and_calculate_intervals(synthetic_data):
    """
    Test the fit_and_calculate_intervals function using precomputed results.
    """
    x, y = synthetic_data

    # Expected results
    expected_popt = [8.24226044, 2.18357947, 0.06424703]
    expected_param_errors = [1.1270266, 0.5212322, 0.0504058]
    expected_r_squared = 0.906
    expected_p_value = 1.11e-16

    # Fit the model and calculate outputs
    popt, param_errors, CI, r_squared, p_value = fit_and_calculate_intervals(x, y, polynomial_model, p0=[1, 1, 1])
    
    # Assertions for parameters
    assert np.allclose(popt, expected_popt, atol=0.01), "Fitted parameters are not within tolerance."
    assert np.allclose(param_errors, expected_param_errors, atol=0.01), "Parameter standard errors are not within tolerance."

    # Assertions for R² and p-value
    assert np.isclose(r_squared, expected_r_squared, atol=0.01), "R² is not within tolerance."
    assert np.isclose(p_value, expected_p_value, atol=1e-16), "p-value is not within tolerance."

    # Assertions for confidence intervals
    assert CI is not None and len(CI) == len(x), "Confidence intervals are not correctly calculated."
