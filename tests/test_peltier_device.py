import unittest
import numpy as np

from src.material import Material
from src.peltier_device import PeltierDevice


class TestPeltierDevice(unittest.TestCase):
    def setUp(self):
        self.n_type = Material(
            name="Test N-type",
            seebeck_coeff=-200e-6,  # V/K
            thermal_conductivity=1.5,  # W/(m·K)
            electrical_resistivity=1e-5,  # Ω·m
            carrier_concentration=1e19
        )

        self.p_type = Material(
            name="Test P-type",
            seebeck_coeff=200e-6,
            thermal_conductivity=1.5,
            electrical_resistivity=1e-5,
            carrier_concentration=1e19
        )

        self.dimensions = (0.01, 0.01, 0.003)  # 1cm x 1cm x 3mm
        self.current = 2.0  # A
        self.device = PeltierDevice(
            n_type=self.n_type,
            p_type=self.p_type,
            dimensions=self.dimensions,
            current=self.current
        )

    def test_initialization(self):
        self.assertEqual(self.device.dimensions, self.dimensions)
        self.assertEqual(self.device.current, self.current)
        self.assertEqual(self.device.n_type, self.n_type)
        self.assertEqual(self.device.p_type, self.p_type)
        self.assertIsNotNone(self.device.heat_model)

    def test_temperature_difference_calculation(self):
        delta_T = self.device.calculate_temperature_difference()

        self.assertGreater(delta_T, 0)

        self.device.current = 0
        delta_T_zero = self.device.calculate_temperature_difference()
        self.assertAlmostEqual(delta_T_zero, 0)

        self.device.current = self.current

    def test_temperature_initialization(self):
        T_cold, T_hot = self.device.initialize_temperature_distribution(
            T_ambient=300)

        # Check that hot side is warmer than cold side
        self.assertGreater(T_hot, T_cold)

        # Check that temperatures are stored correctly
        stored_T_cold, stored_T_hot = self.device.get_temperature_range()
        self.assertEqual(T_cold, stored_T_cold)
        self.assertEqual(T_hot, stored_T_hot)

        # Check that temperature difference matches calculated value
        calculated_delta_T = self.device.calculate_temperature_difference()
        actual_delta_T = T_hot - T_cold
        self.assertAlmostEqual(actual_delta_T, calculated_delta_T, places=10)

    def test_cop_calculation(self):
        # Test with typical temperature difference
        T_hot, T_cold = 300, 280
        cop = self.device.calculate_cop(T_hot, T_cold)

        # COP should be positive for cooling mode
        self.assertGreater(cop, 0)

        # COP should be less than Carnot efficiency
        carnot_cop = T_cold / (T_hot - T_cold)
        self.assertLess(cop, carnot_cop)

        # Test with zero temperature difference
        cop_zero = self.device.calculate_cop(300, 300)
        # COP should be higher with no temperature difference
        self.assertGreater(cop_zero, cop)

        # Test with reversed temperature gradient (heating mode)
        cop_reversed = self.device.calculate_cop(280, 300)
        # COP should be negative for heating mode
        self.assertLess(cop_reversed, 0)

    def test_current_dependence(self):
        """Test dependence of device performance on current"""
        # Test a range of currents
        currents = np.linspace(0, 10, 100)
        delta_Ts = []

        for i in currents:
            self.device.current = i
            delta_Ts.append(self.device.calculate_temperature_difference())

        delta_Ts = np.array(delta_Ts)

        # Find current that gives maximum cooling
        optimal_idx = np.argmax(delta_Ts)
        optimal_current = currents[optimal_idx]

        # Test that there is an optimal current (cooling decreases above this point)
        self.device.current = optimal_current
        delta_T_optimal = self.device.calculate_temperature_difference()

        self.device.current = optimal_current * 2
        delta_T_high = self.device.calculate_temperature_difference()

        # Temperature difference should be lower with current above optimal
        self.assertLess(delta_T_high, delta_T_optimal)

        # Restore original current
        self.device.current = self.current
