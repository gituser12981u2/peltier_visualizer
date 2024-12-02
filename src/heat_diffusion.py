import numpy as np
from typing import Tuple


class HeatDiffusion3D:
    def __init__(self, dimensions: Tuple[float, float, float],
                 thermal_diffusivity: float,
                 dx: float = 0.001,  # 1mm spatial step
                 dt: float = 0.001):  # 0.001s temporal step
        """
        Initialize 3D heat diffusion model.

        Args:
            dimensions: (Lx, Ly, Lz) in meters
            thermal_diffusivity: α = k/(ρCp) in m²/s
            dx: spatial step size in meters
            dt: temporal step size in seconds
        """
        self.dimensions = dimensions
        self.alpha = thermal_diffusivity
        self.dx = dx
        self.dt = dt

        # Calculate grid points
        self.nx = int(dimensions[0]/dx)
        self.ny = int(dimensions[1]/dx)
        self.nz = int(dimensions[2]/dx)

        # Initialize temperature field
        self.T = np.zeros((self.nx, self.ny, self.nz))

        # Check stability condition (von Neumann stability analysis)
        stability_condition = self.alpha * dt / (dx**2)
        if stability_condition > 1/6:  # 3D stability criterion
            raise ValueError(
                f"Unstable configuration. α*dt/dx² = {
                    stability_condition} > 1/6"
            )

    def set_boundary_conditions(self, T_cold: float,
                                T_hot: float, side: str = 'z'):
        """
        Set boundary conditions for the device.

        Args:
            T_cold: Temperature of cold side (K)
            T_hot: Temperature of hot side (K)
            side: Direction of temperature gradient (x, y, or z)
        """
        if side == 'z':
            self.T[:, :, 0] = T_cold
            self.T[:, :, -1] = T_hot
            # Initialize linear temperature gradient
            for k in range(1, self.nz-1):
                self.T[:, :, k] = T_cold + (T_hot - T_cold) * k / (self.nz - 1)
        elif side == 'y':
            self.T[:, 0, :] = T_cold
            self.T[:, -1, :] = T_hot
            for j in range(1, self.ny-1):
                self.T[:, j, :] = T_cold + (T_hot - T_cold) * j / (self.ny - 1)
        elif side == 'x':
            self.T[0, :, :] = T_cold
            self.T[-1, :, :] = T_hot
            for i in range(1, self.nx-1):
                self.T[i, :, :] = T_cold + (T_hot - T_cold) * i / (self.nx - 1)

    def step(self) -> np.ndarray:
        """
        Perform one time step of heat diffusion using finite difference method.
        """
        T_new = np.copy(self.T)
        coeff = self.alpha * self.dt / (self.dx**2)
        print(f"Diffusion coefficient: {coeff:.2e}")

        # Calculate Laplacian with vectorized operations
        T_new[1:-1, 1:-1, 1:-1] = self.T[1:-1, 1:-1, 1:-1] + coeff * (
            # x-direction diffusion
            (self.T[2:, 1:-1, 1:-1] - 2*self.T[1:-1, 1:-1, 1:-1] + self.T[:-2, 1:-1, 1:-1]) +
            # y-direction diffusion
            (self.T[1:-1, 2:, 1:-1] - 2*self.T[1:-1, 1:-1, 1:-1] + self.T[1:-1, :-2, 1:-1]) +
            # z-direction diffusion
            (self.T[1:-1, 1:-1, 2:] - 2*self.T[1:-1,
             1:-1, 1:-1] + self.T[1:-1, 1:-1, :-2])
        )

        # Maintain boundary conditions
        if hasattr(self, 'boundary_side'):
            if self.boundary_side == 'z':
                T_new[:, :, 0] = self.T[:, :, 0]  # Cold side
                T_new[:, :, -1] = self.T[:, :, -1]  # Hot side
            elif self.boundary_side == 'y':
                T_new[:, 0, :] = self.T[:, 0, :]
                T_new[:, -1, :] = self.T[:, -1, :]
            elif self.boundary_side == 'x':
                T_new[0, :, :] = self.T[0, :, :]
                T_new[-1, :, :] = self.T[-1, :, :]

        self.T = T_new
        return self.T

    def get_steady_state(self, tolerance: float = 1e-6,
                         max_iterations: int = 10000) -> np.ndarray:
        """
        Iterate until reaching steady state.

        Args:
            tolerance: Convergence criterion
            max_iterations: Maximum number of iterations

        Returns:
            Steady state temperature field
        """
        iteration = 0
        delta = float('inf')

        while iteration < max_iterations and delta > tolerance:
            T_old = self.T.copy()
            self.step()

            # Check if converged
            delta = np.max(np.abs((self.T - T_old) / T_old))
            iteration += 1

            iteration += 1

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, max change: {delta:.2e}")

        if iteration == max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached")
        else:
            print(f"Converged after {iteration} iterations")

        if iteration == max_iterations:
            print("Maximum iterations reached without convergence")

        return self.T

    def calculate_heat_flux(self, side: str = 'z') -> np.ndarray:
        """
        Calculate heat flux through specified side using central differences.

        Args:
            side: Direction to calculate flux (x, y, or z)

        Returns:
            Heat flux field on specified surface
        """
        if side == 'z':
            # Use central differences for interior points
            dT_dz = np.zeros_like(self.T)
            # Forward difference at cold side
            dT_dz[:, :, 0] = (self.T[:, :, 1] - self.T[:, :, 0]) / self.dx
            # Backward difference at hot side
            dT_dz[:, :, -1] = (self.T[:, :, -1] - self.T[:, :, -2]) / self.dx
            # Central difference for interior points
            dT_dz[:, :, 1:-1] = (self.T[:, :, 2:] -
                                 self.T[:, :, :-2]) / (2 * self.dx)
            return -self.alpha * dT_dz
        elif side == 'y':
            dT_dy = np.zeros_like(self.T)
            dT_dy[:, 0, :] = (self.T[:, 1, :] - self.T[:, 0, :]) / self.dx
            dT_dy[:, -1, :] = (self.T[:, -1, :] - self.T[:, -2, :]) / self.dx
            dT_dy[:, 1:-1, :] = (self.T[:, 2:, :] -
                                 self.T[:, :-2, :]) / (2 * self.dx)
            return -self.alpha * dT_dy
        elif side == 'x':
            dT_dx = np.zeros_like(self.T)
            dT_dx[0, :, :] = (self.T[1, :, :] - self.T[0, :, :]) / self.dx
            dT_dx[-1, :, :] = (self.T[-1, :, :] - self.T[-2, :, :]) / self.dx
            dT_dx[1:-1, :, :] = (self.T[2:, :, :] -
                                 self.T[:-2, :, :]) / (2 * self.dx)
            return -self.alpha * dT_dx
