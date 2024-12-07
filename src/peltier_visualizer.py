from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PeltierVisualizer:
    def __init__(self, peltier_device):
        """
        Initialize visualizer with a PeltierDevice instance

        Args:
            peltier_device: Instance of PeltierDevice class
        """
        self.device = peltier_device
        self.heat_model = peltier_device.heat_model
        self.dimensions = peltier_device.dimensions

    def visualize_temperature_evolution(self, n_frames=100):
        """Create a 3D visualization of temperature distribution evolution"""
        # Get temperature range from device
        T_cold, T_hot = self.device.get_temperature_range()
        print(f"Temperature range: {T_cold:.2f}K to {T_hot:.2f}K")

        # Create mesh grid for visualization
        x = np.linspace(0, self.dimensions[0], self.heat_model.nx)
        y = np.linspace(0, self.dimensions[1], self.heat_model.ny)
        X, Y = np.meshgrid(x, y)

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Set up color normalization around ambient temperature
        vmin, vmax = T_cold - 1, T_hot + 1
        norm = plt.Normalize(vmin, vmax)

        def update(frame):
            ax.clear()

            # Perform several steps per frame for smoother evolution
            for _ in range(5):
                self.heat_model.step()

            # Get the temperature distribution for middle layer
            Z = self.heat_model.T[:, :, self.heat_model.nz//2]

            # Create surface plot
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, norm=norm)

            # Add color bar if it doesn't exist
            if not hasattr(fig, 'colorbar'):
                fig.colorbar = plt.colorbar(surf, ax=ax, shrink=0.5,
                                            aspect=5, label='Temperature (K)')

            # Set labels and title
            ax.set_xlabel('Length (m)')
            ax.set_ylabel('Width (m)')
            ax.set_zlabel('Temperature (K)')
            ax.set_title(f'Temperature Distribution (Frame {frame})')

            # Set consistent view
            ax.view_init(elev=30, azim=45)
            ax.set_zlim(vmin, vmax)

            # Print temperature stats
            print(f"Frame {frame} - Temperature range: \
                  {np.min(Z):.2f}K to {np.max(Z):.2f}K")

            return [surf]

        # Create animation
        anim = FuncAnimation(
            fig, update, frames=n_frames,
            interval=50, blit=True
        )

        return fig, anim

    def visualize_steady_state(self):
        """Create 3D visualization of steady-state temperature distribution"""
        # Ensure device is initialized
        if np.all(self.heat_model.T == 0):
            self.device.initialize_temperature_distribution()

        # Calculate steady state
        self.heat_model.get_steady_state()

        # Create mesh grid for visualization
        x = np.linspace(0, self.dimensions[0], self.heat_model.nx)
        y = np.linspace(0, self.dimensions[1], self.heat_model.ny)
        X, Y = np.meshgrid(x, y)

        # Create figure
        fig = plt.figure(figsize=(15, 10))

        # Create three subplots for different z-planes
        z_positions = [0, self.heat_model.nz//2, -1]
        titles = ['Cold Side', 'Middle Plane', 'Hot Side']

        for i, (z_pos, title) in enumerate(zip(z_positions, titles), 1):
            ax = fig.add_subplot(1, 3, i, projection='3d')
            Z = self.heat_model.T[:, :, z_pos]
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

            ax.set_xlabel('Length (m)')
            ax.set_ylabel('Width (m)')
            ax.set_zlabel('Temperature (K)')
            ax.set_title(f'Temperature Distribution\n{title}')

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        return fig
