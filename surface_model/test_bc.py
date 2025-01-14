import numpy as np
import matplotlib.pyplot as plt
from richards import RichardsEquationSolver
from soil import VanGenuchtenParams
from bc_models import dirichlet_bc, neumann_bc, free_drainage_bc, seepage_bc
from plots import plot_richards_output

def test_bc_seepage_dirichlet():
    """
    Test suite for boundary conditions in Richards Equation Solver.
    """

    # Common parameters
    n_nodes = 50
    z_bottom = -2.0  # [m]
    t_span = (0, 3600)  # Simulate for 1 hour
    t_out = 300  # Output every 5 minutes

    # Van Genuchten parameters for the soil
    vg_params = VanGenuchtenParams(
        theta_r=0.045,
        theta_s=0.43,
        alpha=0.15,
        n=1.56,
        K_s=0.01,
        l=0.5
    )

    # Initial conditions: linear pressure gradient
    h_initial = np.linspace(-5, -1, n_nodes)

    # Test 1: Seepage on top, Dirichlet at bottom
    print("Running Test 1: Seepage on top, Dirichlet at bottom")
    seepage_top_bc = seepage_bc(q_given=0.001, h_crit=-2.0, transition_width=0.5)
    dirichlet_bottom_bc = dirichlet_bc(h_target=-1.0)

    solver_1 = RichardsEquationSolver.from_uniform_mesh(
        n_nodes, z_bottom, vg_params, (seepage_top_bc, dirichlet_bottom_bc)
    )

    result = solver_1.richards_solver(h_initial, t_span, t_out)

    plot_richards_output(result)
    plt.show()

def test_bc_drainage_neumann():
    """
    Test suite for boundary conditions in Richards Equation Solver.
    """

    # Common parameters
    n_nodes = 50
    z_bottom = -2.0  # [m]
    t_span = (0, 3600)  # Simulate for 1 hour
    t_out = 300  # Output every 5 minutes

    # Van Genuchten parameters for the soil
    vg_params = VanGenuchtenParams(
        theta_r=0.045,
        theta_s=0.43,
        alpha=0.15,
        n=1.56,
        K_s=0.01,
        l=0.5
    )

    # Initial conditions: linear pressure gradient
    h_initial = np.linspace(-5, -1, n_nodes)

    # Test 2: Neumann (inflow) on top, Free drainage at bottom
    print("Running Test 2: Neumann (inflow) on top, Free drainage at bottom")
    inflow_top_bc = neumann_bc(q=0.002)
    free_drainage_bottom_bc = free_drainage_bc()

    solver_2 = RichardsEquationSolver.from_uniform_mesh(
        n_nodes, z_bottom, vg_params, (inflow_top_bc, free_drainage_bottom_bc)
    )

    times, results = solver_2.richards_solver(h_initial, t_span, t_out)

    # Plot results for Test 2
    plt.figure()
    for i, h in enumerate(results.T):
        plt.plot(np.linspace(0, z_bottom, n_nodes), h, label=f"t={times[i]:.0f}s")
    plt.title("Test 2: Inflow on top, Free drainage at bottom")
    plt.xlabel("Depth (m)")
    plt.ylabel("Pressure Head (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_bc_seepage_dirichlet()
    test_bc_drainage_neumann()
