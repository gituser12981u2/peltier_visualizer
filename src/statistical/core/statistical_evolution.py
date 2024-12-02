from abc import ABC, abstractmethod
import numpy as np
from scipy import constants as const
from typing import Dict, Tuple

from src.statistical.core.bosonic.boson_statistics import \
    BoseEinsteinStatistics
from src.statistical.core.fermionic.fermion_statistics import \
    FermiDiracStatistics
from src.statistical.core.particle_interactions import ParticleInteractions
from src.statistical.thermal.debye_model import DebyeModel
from src.statistical.thermal.thermal_properties import ThermalProperties


class StatisticalSystem(ABC):
    """Abstract base class for quantum statistical systems"""

    def __init__(self):
        # Physical constants
        self.kb = const.k
        self.hbar = const.hbar
        self.q = const.e

        # Cache for expensive calculations
        self._cache: Dict[str, Dict[Tuple, float]] = {}

    def _clear_cache(self):
        """Clear cached values when system parameters change significantly"""
        self._cache.clear()

    @abstractmethod
    def update_temperature(self, T: float):
        """Update system temperature"""
        pass


class FermionicEvolution(StatisticalSystem):
    """Handles Fermi-Dirac statistics with temperature evolution"""

    def __init__(self, statistics: FermiDiracStatistics):
        super().__init__()
        self.statistics = statistics
        self._initial_n = self.statistics.calculate_carrier_density('electron')
        self._initial_p = self.statistics.calculate_carrier_density('hole')
        self._initial_temperature = statistics.config.temperature
        self.is_degenerate = False

    def update_temperature(self, T: float):
        """Update system temperature maintaining carrier density"""
        if abs(T - self.statistics.config.temperature) > 0.1:
            self._clear_cache()

            # Calculate band gap at new temperature
            band_gap = self.statistics._calculate_band_gap(T)
            self.statistics.config.temperature = T
            self.statistics.kT = self.statistics.kb_eV * T
            self.statistics.config.conduction_band_edge = 0
            self.statistics.config.valence_band_edge = -band_gap

            # Update temperature-dependent parameters
            initial_excess = self._initial_n - self._initial_p

            def charge_balance(E_f):
                self.statistics.config.fermi_level = E_f
                n = self.statistics.calculate_carrier_density('electron')
                p = self.statistics.calculate_carrier_density('hole')
                current_excess = n - p
                return current_excess - initial_excess

            # Estimate new Fermi level position
            intrinsic_level = -band_gap/2
            thermal_voltage = self.statistics.kT

            # Find new Fermi level using root finding
            from scipy.optimize import root_scalar
            try:
                result = root_scalar(
                    charge_balance,
                    x0=intrinsic_level,
                    x1=intrinsic_level + thermal_voltage,
                    method='secant',
                    maxiter=100
                )
                self.statistics.config.fermi_level = result.root
            except ValueError:
                # If root finding fails, estimate based on temperature scaling
                self.statistics.config.fermi_level = intrinsic_level + \
                    (self.statistics.config.fermi_level
                     - (-self.statistics._calculate_band_gap(
                         self._initial_temperature)/2)) * \
                    (T/self._initial_temperature)

    def calculate_screening_wavevector(self) -> float:
        """Calculate Thomas-Fermi screening wavevector"""
        cache_key = ('screening', self.statistics.config.temperature)
        if cache_key in self._cache:
            return self._cache[cache_key]

        n = self.statistics.calculate_carrier_density()
        E_f = abs(self.statistics.config.fermi_level)
        kT = self.kb * self.statistics.config.temperature

        epsilon = self.statistics.calculate_dielectric_constant(
            self.statistics.config.temperature)

        # Thomas-Fermi screening
        if self.is_degenerate:
            # Use degenerate electron gas formula
            qTF = np.sqrt(6 * np.pi * n * self.q**2 /
                          (epsilon * const.epsilon_0 * E_f))
        else:
            # Use non-degenerate formula with temperature correction
            qTF = np.sqrt(self.q**2 * n /
                          (epsilon * const.epsilon_0 * kT))

        self._cache[cache_key] = qTF
        return qTF


class BosoniccEvolution(StatisticalSystem):
    """Handles evolution of Bose-Einstein systems"""

    def __init__(self,
                 statistics: BoseEinsteinStatistics,
                 debye_model: DebyeModel,
                 thermal_props: ThermalProperties):
        super().__init__()
        self.statistics = statistics
        self.debye_model = debye_model
        self.thermal_props = thermal_props

    def update_temperature(self, T: float):
        """Update system temperature"""
        if abs(T - self.statistics.config.temperature) > 0.1:
            self._clear_cache()
            self.statistics.config.temperature = T

            # Update thermal properties
            self.thermal_props.calculate_specific_heat(T)
            self.thermal_props.calculate_thermal_conductivity(T)

    def calculate_total_population(self) -> float:
        """Calculate total phonon population with caching"""
        cache_key = ('total_population', self.statistics.config.temperature)
        if cache_key in self._cache:
            return self._cache[cache_key]

        population = self.statistics.calculate_total_population()
        self._cache[cache_key] = population
        return population


class ParticleEvolution:
    """Handles coupled evolution of fermion and boson systems"""

    def __init__(self,
                 fermion_system: FermionicEvolution,
                 boson_system: BosoniccEvolution):
        self.fermions = fermion_system
        self.bosons = boson_system

        # Initialize particle interactions
        self.interactions = ParticleInteractions(
            self.fermions.statistics,
            self.bosons.statistics,
            self.bosons.debye_model
        )

    def calculate_average_scattering_rate(self, T: float) -> float:
        """Calculate thermally averaged scattering rate"""
        # Get energy range for averaging (few kT around Fermi level)
        E_f = self.fermions.statistics.config.fermi_level
        kT = self.fermions.statistics.kT

        # Sample energies around Fermi level
        energies = np.linspace(E_f - 5*kT, E_f + 5*kT, 20)

        # Calculate Fermi-Dirac weighted average of scattering rates
        total_weight = 0
        total_rate = 0

        for E in energies:
            # Get occupation at this energy
            weight = self.fermions.statistics.fermi_dirac(E)

            # Calculate scattering rate at this energy
            rates = self.interactions.calculate_scattering_rate(
                E, T, self.bosons.debye_model.config.acoustic_deformation_potential
            )

            total_rate += weight * rates['total_rate']
            total_weight += weight

        return total_rate / total_weight if total_weight > 0 else 0

    def evolve_temperature(self, T_initial: float,
                           T_final: float,
                           num_steps: int = 100) -> Dict[str, np.ndarray]:
        """Evolve coupled system through temperature range"""
        temperatures = np.linspace(T_initial, T_final, num_steps)
        results = {
            'temperature': temperatures,
            'chemical_potential': np.zeros(num_steps),
            'carrier_density': np.zeros(num_steps),
            'screening_length': np.zeros(num_steps),
            'phonon_population': np.zeros(num_steps),
            'thermal_conductivity': np.zeros(num_steps),
            'specific_heat': np.zeros(num_steps),
            'scattering_rate': np.zeros(num_steps),
            'emission_rate': np.zeros(num_steps),
            'absorption_rate': np.zeros(num_steps)
        }

        for i, T in enumerate(temperatures):
            # Update both systems
            self.fermions.update_temperature(T)
            self.bosons.update_temperature(T)

            # Calculate scattering rates at Fermi level
            scattering_rates = self.interactions.calculate_scattering_rate(
                self.fermions.statistics.config.fermi_level,
                T,
                self.bosons.debye_model.config.acoustic_deformation_potential
            )

            # Record results
            results['chemical_potential'][i] = self.fermions.statistics.config.fermi_level
            results['carrier_density'][i] = self.fermions.statistics.calculate_carrier_density()
            results['screening_length'][i] = 2*np.pi / \
                self.fermions.calculate_screening_wavevector()
            results['phonon_population'][i] = self.bosons.statistics.calculate_total_population()
            results['thermal_conductivity'][i] = self.bosons.thermal_props.calculate_thermal_conductivity(
                T)
            results['specific_heat'][i] = self.bosons.thermal_props.calculate_specific_heat(
                T)
            results['scattering_rate'][i] = scattering_rates['total_rate']
            results['emission_rate'][i] = scattering_rates['emission_rate']
            results['absorption_rate'][i] = scattering_rates['absorption_rate']

        return results
