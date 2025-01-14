import numpy as np

def dirichlet_bc(h_target):
    """
    Factory for Dirichlet boundary condition applied in a weak sense.
    Returns a function that calculates the inward flux.

    Parameters:
    h_target : float
        The target pressure head for the boundary.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        return -K * (H - h_target) / dz

    return bc_function

def neumann_bc(q):
    """
    Factory for Neumann boundary condition (given flux).
    Returns a function that calculates the inward flux.

    Parameters:
    q : float
        Prescribed water flux.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        return q

    return bc_function

def free_drainage_bc():
    """
    Factory for free drainage boundary condition.
    Returns a function that calculates the inward flux.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        return K

    return bc_function

def seepage_bc(q_given, h_crit, transition_width):
    """
    Factory for seepage boundary condition.
    Smooth interpolation between given flux and pressure.

    Parameters:
    q_given : float
        Prescribed flux.
    h_crit : float
        Critical pressure head.
    transition_width : float
        Width of the pressure head transition region for interpolation.

    Returns:
    callable
        A function that computes inward water flux.
    """
    def bc_function(t, H, K, dz):
        factor = np.tanh((H - h_crit) / transition_width) * 0.5 + 0.5
        dirichlet_term = -K * (H - h_crit) / dz
        prescribed_flux = q_given
        return factor * dirichlet_term + (1 - factor) * prescribed_flux

    return bc_function
