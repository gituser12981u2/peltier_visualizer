from src.heat_diffusion import HeatDiffusion3D


class PeltierDevice:
    def __init__(self, n_type, p_type, dimensions, current=1.0):
        """
        Initialize a Peltier device with given materials and dimensions

        Args:
            n_type: N-type semiconductor material
            p_type: P-type semiconductor material
            dimensions: (length, width, height) in meters
            current: Operating current in Amperes
        """
        self.n_type = n_type
        self.p_type = p_type
        self.dimensions = dimensions  # (length, width, height) in meters
        self.current = current  # Amps

        # Material properties for Bi2Te3
        # TODO: make these inputted values by client code to abstract away from peltier device
        self.density = 7700  # kg/m^3
        self.specific_heat = 544  # J(kg*K)

        # Initialize heat diffusion model
        # Calculate effective thermal diffusivity from material properties
        alpha_n = n_type.thermal_conductivity / \
            (self.density * self.specific_heat)
        alpha_p = p_type.thermal_conductivity / \
            (self.density * self.specific_heat)
        alpha_eff = (alpha_n + alpha_p) / 2  # Effective diffusivity

        self.heat_model = HeatDiffusion3D(dimensions, alpha_eff)
        self._T_cold = None
        self._T_hot = None

    def calculate_temperature_difference(self):
        """Calculate steady-state temperature distribution"""
        seebeck_total = abs(self.n_type.seebeck_coeff) + \
            abs(self.p_type.seebeck_coeff)

        # Calculate device resistance
        R = (self.n_type.electrical_resistivity +
             self.p_type.electrical_resistivity) * \
            self.dimensions[2] / (self.dimensions[0] * self.dimensions[1])

        # Calculate thermal conductance
        K = (self.n_type.thermal_conductivity +
             self.p_type.thermal_conductivity) * \
            (self.dimensions[0] * self.dimensions[1]) / self.dimensions[2]

        # Calculate temperature difference using simplified model
        delta_T = (seebeck_total * self.current *
                   300 - 0.5 * self.current**2 * R) / K
        print(f"Delta T: {delta_T} K")
        return delta_T

    # TODO: make T_ambient given by client code
    def initialize_temperature_distribution(self, T_ambient=300):
        """Initialize temperature distribution with calculated temperature difference"""
        delta_T = self.calculate_temperature_difference()
        self._T_cold = T_ambient - delta_T/2
        self._T_hot = T_ambient + delta_T/2

        # Set initial conditions
        self.heat_model.set_boundary_conditions(
            self._T_cold, self._T_hot, side='z')
        return self._T_cold, self._T_hot

    def get_temperature_range(self):
        """Get the current temperature range of the device"""
        if self._T_cold is None or self._T_hot is None:
            self.initialize_temperature_distribution()
        return self._T_cold, self._T_hot

    def calculate_cop(self, T_hot, T_cold):
        """Calculate Coefficient of Performance with realistic physics"""
        delta_T = T_hot - T_cold

        # Electrical resistance of the device
        R = (self.n_type.electrical_resistivity +
             self.p_type.electrical_resistivity) * \
            self.dimensions[2] / (self.dimensions[0] * self.dimensions[1])

        # Thermal conductance
        K = (self.n_type.thermal_conductivity +
             self.p_type.thermal_conductivity) * \
            (self.dimensions[0] * self.dimensions[1]) / self.dimensions[2]

        # Seebeck coefficient of the device
        seebeck_total = abs(self.n_type.seebeck_coeff) + \
            abs(self.p_type.seebeck_coeff)

        # Cooling power components
        Q_peltier = seebeck_total * self.current * T_cold  # Peltier cooling
        Q_joule = 0.5 * self.current**2 * R  # Half of Joule heating goes to cold side
        Q_cond = K * delta_T  # Heat conduction from hot to cold

        cooling_power = Q_peltier - Q_joule - Q_cond

        # Input power components
        P_joule = self.current**2 * R  # Joule heating
        P_seebeck = seebeck_total * self.current * \
            delta_T  # Power against Seebeck voltage

        power_input = P_joule + P_seebeck

        return cooling_power / power_input
