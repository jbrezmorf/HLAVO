import numpy as np
import pytest
import matplotlib.pyplot as plt
from plots import covariance_plot

def generate_random_cov_matrix(dim, seed=42):
    """
    Generates a random symmetric positive semi-definite covariance matrix.
    """
    np.random.seed(seed)
    A = np.random.randn(dim, dim)
    cov_matrix = np.dot(A, A.T)  # Ensures positive semi-definiteness
    return cov_matrix

def test_plot_covariance():
    """
    Test if plot_covariance_analysis runs without errors.
    """
    # Generate a 5x5 random covariance matrix
    cov_matrix = generate_random_cov_matrix(dim=5)

    # Set K (number of eigenvectors to plot)
    K = 3

    # Run the plotting function inside a try-except block
    covariance_plot(cov_matrix, 0.1, K)
    plt.close('all')  # Close plots after testing
