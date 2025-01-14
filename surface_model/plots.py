import matplotlib.pyplot as plt
import numpy as np

def plot_richards_output(output):
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
    z_nodes = np.linspace(0, 1, H.shape[1])  # Assuming a normalized vertical domain

    # Prepare the Z coordinate indices for observation
    z_indices = [np.argmin(np.abs(z_nodes - z)) for z in z_coords]

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Top-left: Color plot of H vs time and depth
    c1 = axs[0, 0].imshow(H.T, aspect='auto', extent=[times[0], times[-1], z_nodes[0], z_nodes[-1]],
                          origin='lower', cmap='viridis')
    fig.colorbar(c1, ax=axs[0, 0])
    axs[0, 0].set_title('Pressure Head (H)')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Depth')

    # Top-right: Color plot of Moisture vs time and depth
    c2 = axs[0, 1].imshow(moisture.T, aspect='auto', extent=[times[0], times[-1], z_nodes[0], z_nodes[-1]],
                           origin='lower', cmap='viridis')
    fig.colorbar(c2, ax=axs[0, 1])
    axs[0, 1].set_title('Moisture Content')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Depth')

    # Middle-left: Line plot of H at selected observation points
    for idx in z_indices:
        axs[1, 0].plot(times, H[:, idx], label=f'Depth={z_nodes[idx]:.2f}')
    axs[1, 0].legend()
    axs[1, 0].set_title('Pressure Head (H) at Observation Points')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Pressure Head (H)')

    # Middle-right: Line plot of Moisture at selected observation points
    for idx in z_indices:
        axs[1, 1].plot(times, moisture[:, idx], label=f'Depth={z_nodes[idx]:.2f}')
    axs[1, 1].legend()
    axs[1, 1].set_title('Moisture at Observation Points')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Moisture Content')

    # Bottom row: Line plot of top and bottom fluxes
    axs[2, 0].plot(times, top_flux, label='Top Flux')
    axs[2, 0].plot(times, bottom_flux, label='Bottom Flux')
    axs[2, 0].legend()
    axs[2, 0].set_title('Boundary Fluxes')
    axs[2, 0].set_xlabel('Time')
    axs[2, 0].set_ylabel('Flux')

    axs[2, 1].axis('off')  # Leave the last subplot blank

    plt.tight_layout()
    plt.show()
