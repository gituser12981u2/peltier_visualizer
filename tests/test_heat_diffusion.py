import unittest
import numpy as np

from src.heat_diffusion import HeatDiffusion3D


class TestHeatDiffusion3D(unittest.TestCase):
    def setUp(self):
        self.dimensions = (0.01, 0.01, 0.003)  # 1cm x 1cm x 3mm
        self.alpha = 1e-6  # Typical for semiconductors
        self.model = HeatDiffusion3D(self.dimensions, self.alpha)

    def test_initialization(self):
        self.assertEqual(self.model.T.shape, (10, 10, 3))
        self.assertTrue(np.allclose(self.model.T, 0))  # Initially zero

    def test_boundary_conditions(self):
        T_cold, T_hot = 280, 300
        self.model.set_boundary_conditions(T_cold, T_hot)

        # Check cold side
        self.assertTrue(np.allclose(self.model.T[:, :, 0], T_cold))
        # Check hot side
        self.assertTrue(np.allclose(self.model.T[:, :, -1], T_hot))

    def test_heat_flux_conservation(self):
        T_cold, T_hot = 280, 300
        self.model.set_boundary_conditions(T_cold, T_hot)
        self.model.get_steady_state()

        # In steady state, heat flux should be constant through any z-plane
        flux = self.model.calculate_heat_flux('z')
        mean_flux_cold = np.mean(flux[:, :, 0])
        mean_flux_hot = np.mean(flux[:, :, -1])

        self.assertAlmostEqual(abs(mean_flux_cold),
                               abs(mean_flux_hot), places=5)

    def test_steady_state_linear_gradient(self):
        T_cold, T_hot = 280, 300
        self.model.set_boundary_conditions(T_cold, T_hot)
        self.model.get_steady_state()

        # Check middle point temperature
        mid_temp = self.model.T[5, 5, 1]
        expected_mid_temp = (T_hot + T_cold) / 2

        self.assertAlmostEqual(mid_temp, expected_mid_temp, places=1)

    def test_step_evolution(self):
        """Test that step() actually evolves the temperature field"""
        nx, ny, nz = self.model.nx, self.model.ny, self.model.nz
        center_x, center_y = nx//2, ny//2
        radius = 1

        T_cold = 300.0
        T_hot = 320.0
        self.model.set_boundary_conditions(T_cold, T_hot, side='z')
        self.model.boundary_side = 'z'

        z_mid = nz//2
        expected_mid_temp = T_cold + \
            (T_hot - T_cold) * z_mid / (self.model.nz - 1)
        hot_spot_temp = expected_mid_temp + 10

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < radius:
                    self.model.T[i, j, z_mid] = hot_spot_temp
                elif dist < radius + 1:
                    frac = radius + 1 - dist
                    self.model.T[i, j, z_mid] = expected_mid_temp + \
                        (hot_spot_temp - expected_mid_temp) * frac

        hot_spot_state = self.model.T.copy()
        initial_energy = np.sum(hot_spot_state)

        def get_mid_layer_max():
            return np.max(self.model.T[1:-1, 1:-1, z_mid])

        initial_max = get_mid_layer_max()

        n_steps = 10
        for step in range(n_steps):
            initial_layer = self.model.T[:, :,
                                         z_mid].copy()
            self.model.step()
            current_energy = np.sum(self.model.T)
            current_max = get_mid_layer_max()

            print(f"\nStep {step+1} diagnostics:")
            print(f"Initial energy: {initial_energy:.6f}")
            print(f"Current energy: {current_energy:.6f}")
            print(f"Energy difference: {current_energy - initial_energy:.6f}")
            print(f"Max temperature in middle layer: {current_max:.6f}")
            print(f"Temperature change in middle layer: {
                  np.max(self.model.T[:, :, z_mid] - initial_layer):.6f}")

            if step > 0:
                self.assertLess(
                    current_max,
                    initial_max,
                    f"Middle layer maximum temperature should decrease due to diffusion at step {
                        step+1}"
                )

    def test_step_boundary_conditions(self):
        """Test that step() maintains boundary conditions"""
        T_cold, T_hot = 280, 300
        self.model.set_boundary_conditions(T_cold, T_hot, side='z')

        # Store initial boundary values
        initial_cold = self.model.T[:, :, 0].copy()
        initial_hot = self.model.T[:, :, -1].copy()

        # Perform several steps
        n_steps = 10
        for i in range(n_steps):
            self.model.step()

            # Check that boundaries remain unchanged
            self.assertTrue(np.allclose(self.model.T[:, :, 0], initial_cold),
                            f"Cold boundary changed at step {i+1}")
            self.assertTrue(np.allclose(self.model.T[:, :, -1], initial_hot),
                            f"Hot boundary changed at step {i+1}")

            print(f"Step {i+1} boundaries ok - Cold: {np.mean(self.model.T[:, :, 0]):.2f}K, "
                  f"Hot: {np.mean(self.model.T[:, :, -1]):.2f}K")
