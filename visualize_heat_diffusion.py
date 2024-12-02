import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.heat_diffusion import HeatDiffusion3D


def visualize_heat_diffusion():
    # Initialize heat diffusion model
    dimensions = (0.01, 0.01, 0.003)  # 1cm x 1cm x 3mm
    alpha = 1e-6  # Typical thermal diffusivity for semiconductors (m²/s)
    model = HeatDiffusion3D(dimensions, alpha)

    # Set uniform initial temperature
    T_ambient = 300  # K
    model.T.fill(T_ambient)

    # Create a hot spot in the center
    nx, ny, nz = model.nx, model.ny, model.nz
    center_x, center_y = nx//2, ny//2
    radius = nx//4  # Size of hot spot

    # Add hot spot in middle layer
    z_mid = nz//2
    hot_spot_temp = T_ambient + 20  # 20K above ambient

    for i in range(nx):
        for j in range(ny):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < radius:
                model.T[i, j, z_mid] = hot_spot_temp
            elif dist < radius + 2:
                # Create smooth transition at boundary
                frac = 1 - (dist - radius)/2
                model.T[i, j, z_mid] = T_ambient + \
                    (hot_spot_temp - T_ambient) * frac

    # Create figure for visualization
    fig = plt.figure(figsize=(15, 5))

    # Create three subplots for different z-planes
    axs = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
    z_positions = [0, z_mid, -1]
    titles = ['Bottom Layer', 'Middle Layer (Hot Spot)', 'Top Layer']

    # Create mesh grid for visualization
    x = np.linspace(0, dimensions[0], nx)
    y = np.linspace(0, dimensions[1], ny)
    X, Y = np.meshgrid(x, y)

    # Set up color normalization
    vmin, vmax = T_ambient - 1, hot_spot_temp + 1

    # Initialize plots
    ims = []
    for ax, z_pos, title in zip(axs, z_positions, titles):
        im = ax.pcolormesh(X, Y, model.T[:, :, z_pos].T,
                           cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ims.append(im)

    # Add colorbar
    plt.colorbar(ims[0], ax=axs, label='Temperature (K)')

    def update(frame):
        # Perform several steps per frame for smoother evolution
        for _ in range(5):
            model.step()

        # Update plots
        for im, z_pos in zip(ims, z_positions):
            im.set_array(model.T[:, :, z_pos].T)

        # Print temperature stats
        print(f"Frame {frame} - Temperature range: {np.min(model.T)
              :.2f}K to {np.max(model.T):.2f}K")
        return ims

    # Create animation
    anim = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    plt.tight_layout()

    return fig, anim


if __name__ == "__main__":
    fig, anim = visualize_heat_diffusion()
    plt.show()
