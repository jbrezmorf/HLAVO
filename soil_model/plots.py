import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import attrs

@attrs.define
class RichardsSolverOutput:
    """
    Output class for Richards Equation Solver.

    Attributes:
    ----------
    times : np.ndarray
        Array of output times.
    H : np.ndarray, times in rows.
        Pressure head array at output times.
    moisture : np.ndarray
        Moisture content (absolute saturation) at output times.
    top_flux : np.ndarray
        Influx at the top boundary at output times.
    bottom_flux : np.ndarray
        Influx at the bottom boundary at output times.
    """
    times: np.ndarray
    H: np.ndarray
    moisture: np.ndarray
    top_flux: np.ndarray
    bottom_flux: np.ndarray
    nodes_z: np.ndarray



def plot_richards_output(output, obs_points=None, fname=None, show=False):
    """
    Plot the results from RichardsSolverOutput.

    Parameters:
    -----------
    output : RichardsSolverOutput
        The solver output containing times, pressure head (H), moisture, and fluxes.
    z_coords : list
        List of Z coordinates for observation points to include in the line plots.
    """
    times = output.times
    z_coords = output.nodes_z
    H = output.H
    moisture = output.moisture
    top_flux = output.top_flux
    bottom_flux = output.bottom_flux
    if obs_points is None:
        step = int(len(z_coords) / 10)    # 10 obs. points
        obs_points = z_coords[0:None:step]

    # Prepare the Z coordinate indices for observation
    i_obs_points = [np.argmin(np.abs(z_obs - z_coords)) for z_obs in obs_points]

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))


    # Top-left: Color plot of H vs time and depth
    c1 = axs[0, 0].imshow(H.T, aspect='auto', extent=[times[0], times[-1], z_coords[0], z_coords[-1]],
                          origin='lower', cmap=plt.cm.viridis_r)
    fig.colorbar(c1, ax=axs[0, 0])
    axs[0, 0].set_title('Pressure Head (H)')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Depth')

    # Top-right: Color plot of Moisture vs time and depth
    # Use the reversed viridis colormap
    sat_cmap = plt.cm.viridis_r  # reversed viridis
    sat_norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)  # linear normalization

    c2 = axs[0, 1].imshow(moisture.T, aspect='auto', extent=[times[0], times[-1], z_coords[0], z_coords[-1]],
                           origin='lower', cmap=sat_cmap, norm=sat_norm)
    fig.colorbar(c2, ax=axs[0, 1])
    axs[0, 1].set_title('Moisture Content')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Depth')

    # Middle-left: Line plot of H at selected observation points
    for idx in i_obs_points:
        axs[1, 0].plot(times, H[:, idx], label=f'z={z_coords[idx]:.2f}')
    axs[1, 0].legend()
    axs[1, 0].set_title('Pressure Head (H) at Observation Points')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Pressure Head (H)')

    # Middle-right: Line plot of Moisture at selected observation points
    for idx in i_obs_points:
        axs[1, 1].plot(times, moisture[:, idx], label=f'Depth={z_coords[idx]:.2f}')
    axs[1, 1].legend()
    axs[1, 1].set_title('Moisture at Observation Points')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Moisture Content')

    if not (top_flux is None or bottom_flux is None):
        # Bottom row: Line plot of top and bottom fluxes
        axs[2, 0].plot(times, top_flux, label='Top Flux')
        axs[2, 0].plot(times, bottom_flux, label='Bottom Flux')
        axs[2, 0].legend()
        axs[2, 0].set_title('Boundary Fluxes')
        axs[2, 0].set_xlabel('Time')
        axs[2, 0].set_ylabel('Flux')

    axs[2, 1].axis('off')  # Leave the last subplot blank

    fig.tight_layout()
    if fname is None:
        fname = 'richards_solver_output.pdf'
    fig.savefig(fname)
    if show:
        plt.show()

def covariance_plot(cov_matrix, time, n_evec=10, fname=None, show=False):
    """
    Plot the covariance matrix as a heatmap.

    Parameters:
    -----------
    covariance : np.ndarray
        The covariance matrix to plot.
    title : str
        Title for the plot.
    """

    """
    Plots analysis of a given covariance matrix:
    1. Variances (log scale)
    2. Correlation matrix heatmap (signed log scale colormap)
    3. Sorted eigenvalues
    4. First K eigenvectors as a function of index i

    Parameters:
        cov_matrix (numpy.ndarray): The covariance matrix.
        K (int): Number of eigenvectors to plot.
    """
    title = f'Covariance Analysis at time {time}'
    if fname is None:
        fname = f'covariance_analysis_{time}.pdf'
    # Compute variances (diagonal elements)
    variances = np.diag(cov_matrix)

    # Compute correlation matrix
    std_dev = np.sqrt(variances)
    correlation_matrix = cov_matrix / np.outer(std_dev, std_dev)

    # Compute Eigenvalues and Eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # Ensures sorted order (ascending)
    eigvals_sorted = eigvals[::-1]  # Descending order
    eigvecs_sorted = eigvecs[:, ::-1]  # Reorder eigenvectors accordingly

    # Set up figure layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Variance bar plot (log scale)
    axes[0, 0].bar(range(len(variances)), variances, color='blue', alpha=0.7)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Variances (Log Scale)')
    axes[0, 0].set_xlabel('Component Index')
    axes[0, 0].set_ylabel('Variance (log scale)')

    # 2. Correlation matrix heatmap (signed log scale colormap)
    signed_log_corr = np.sign(correlation_matrix) * np.log1p(np.abs(correlation_matrix))
    sns.heatmap(signed_log_corr, ax=axes[0, 1], center=0, cmap='coolwarm', annot=False)
    axes[0, 1].set_title('Correlation Matrix (Signed Log Scale)')

    # 3. Sorted Eigenvalues plot
    axes[1, 0].plot(eigvals_sorted, 'o-', color='red', markersize=5)
    axes[1, 0].set_title('Sorted Eigenvalues')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Eigenvalue')

    # 4. First K Eigenvectors as functions of index i
    for i in range(min(n_evec, eigvecs_sorted.shape[1])):
        axes[1, 1].plot(eigvecs_sorted[:, i], label=f'Eigenvector {i + 1}')

    axes[1, 1].set_title(f'First {n_evec} Eigenvectors')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Component Value')
    axes[1, 1].legend()

    fig.tight_layout()
    fig.suptitle(title)
    fig.savefig(fname)
    if show:
        plt.show()