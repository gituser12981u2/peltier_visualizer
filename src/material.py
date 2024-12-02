class Material:
    def __init__(self, name, seebeck_coeff, thermal_conductivity,
                 electrical_resistivity, carrier_concentration,
                 temperature=300):
        self.name = name
        self.seebeck_coeff = seebeck_coeff  # V/K
        self.thermal_conductivity = thermal_conductivity  # W/(m*K)
        self.electrical_resistivity = electrical_resistivity  # ohm * m
        self.carrier_concentration = carrier_concentration  # carriers/m^3
        self.temperature = temperature  # K

    def calculate_zt(self, T):
        """Calculate the dimensionless figure of merit ZT"""
        electrical_conductivity = 1 / self.electrical_resistivity
        power_factor = self.seebeck_coeff**2 * electrical_conductivity
        zt = power_factor * T / self.thermal_conductivity
        return zt

    def update_properties_with_temperature(self, T):
        """Update material properties based on temperature dependence"""
        # To be implemented with actual temperature dependencies
        pass
