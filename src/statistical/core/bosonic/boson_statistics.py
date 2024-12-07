from dataclasses import dataclass
import numpy as np
from typing import Union
from scipy import constants as const


@dataclass
class BoseEinsteinConfig:
    """Configuration parameters for Bose-Einstein statistics calculations"""
    temperature: float  # Temperature in Kelvin
    cutoff_energy: float  # Maximum phonon energy in eV
    lattice_constant: float  # Lattice constant in meters
    sound_velocity: float  # Speed of sound in material (m/s)
    chemical_potential: float = 0.0  # Chemical potential in eV (0 for phonons)
    dispersion_type: str = 'debye'  # Type of dispersion relation
    branch_degeneracy: int = 3  # Number of phonon branches


class BoseEinsteinStatistics:
    """
    Class implementing Bose-Einstein statistics for phonons.

    This class handles calculations of phonon populations,
    Bose-Einstein distributions, and related quantities.
    """

    def __init__(self, config: BoseEinsteinConfig):
        """
        Initialize BoseEinsteinStatistics with material parameters.

        Args:
            config: Configuration parameters for calculations
        """
        self.config = config

        # Set up fundamental constants
        self.kb = const.k  # Boltzmann constant
        self.kb_eV = const.k / const.e  # Boltzmann constant in eV/K
        self.h = const.h  # Planck constant
        self.hbar = const.hbar  # Reduced Planck constant
        self.q = const.e  # Elementary charge

        # Thermal energy in eV
        self.kT = self.kb_eV * self.config.temperature

    def bose_einstein(self, E: Union[float,
                                     np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the Bose-Einstein distribution function.

        Args:
            E: Energy level in eV

        Returns:
            Bose-Einstein occupation probability
        """
        # Handle potential overflow in exp
        eta = (E - self.config.chemical_potential) / self.kT

        # Use exponential form for numerical stability
        with np.errstate(over='ignore', divide='ignore'):
            return 1 / (np.exp(eta) - 1)

    def density_of_states_3D(self, E: float) -> float:
        """
        Calculate the 3D phonon density of states using Debye model

        Args:
            E: energy in eV

        Returns:
            Density of states in states/(eV*m^3)
        """
        if E <= 1e-10 or E > self.config.cutoff_energy:
            return 0.0

        # Calculate with Debye model
        prefactor = self.config.branch_degeneracy * 3 / (2 * np.pi**2)
        prefactor *= (1 / self.config.sound_velocity**3)

        # Convert energy to angular frequency
        omega = E * self.q / self.hbar

        return prefactor * omega**2

    def calculate_phonon_population(self, E: float) -> float:
        """
        Calculate phonon population at given energy.

        Args:
            E: Phonon energy in eV

        Returns:
            Number of phonons per unit volume (1/m^3)
        """
        if E < 1e-10 or E > self.config.cutoff_energy:
            return 0.0

        try:
            dos = self.density_of_states_3D(E)
            occupation = self.bose_einstein(E)
            population = dos * occupation
            return population if not np.isnan(population) else 0.0
        except (RuntimeWarning, RuntimeError):
            return 0.0

    def calculate_total_population(self) -> float:
        """
        Calculate total phonon population by numerical integration.

        Returns:
            Total phonon population per unit volume (1/m^3)
        """
        def integrand(E):
            return self.calculate_phonon_population(E)

        # Integration range from near-zero to cuttoff energy
        E_low = np.linspace(13-6, 0.01, 200)  # Dense sampling for low energies
        E_high = np.linspace(0.01, self.config.cutoff_energy, 800)
        E_points = np.concatenate([E_low, E_high])

        # Numerical integration using trapezoidal rule
        populations = []
        for E in E_points:
            try:
                pop = integrand(E)
                populations.append(pop if not np.isnan(pop) else 0.0)
            except (RuntimeWarning, RuntimeError):
                populations.append(0.0)

        total_pop = np.trapz(populations, E_points)
        return total_pop if not np.isnan(total_pop) else 0.0

    def calculate_average_energy(self) -> float:
        """
        Calculate average phonon energy.

        Returns:
            Average phonon energy in eV
        """
        def energy_integrand(E):
            try:
                return E * self.calculate_phonon_population(E)
            except (RuntimeWarning, RuntimeError):
                return 0.0

        # Use same energy sampling as total population
        E_low = np.linspace(1e-6, 0.01, 200)
        E_high = np.linspace(0.01, self.config.cutoff_energy, 800)
        E_points = np.concatenate([E_low, E_high])

        energy_weighted = []
        for E in E_points:
            val = energy_integrand(E)
            energy_weighted.append(val if not np.isnan(val) else 0.0)

        total_energy = np.trapz(energy_weighted, E_points)
        total_population = self.calculate_total_population()

        if total_population > 0:
            avg_energy = total_energy / total_population
            return avg_energy if not np.isnan(avg_energy) else 0.0
        return 0.0

    def phonon_summary(self) -> dict:
        """
        Generate a summary of phonon statistics.

        Returns:
            Dictionary containing key phonon statistics
        """
        return {
            'temperature': self.config.temperature,
            'total_population': self.calculate_total_population(),
            'average_energy': self.calculate_average_energy(),
            'cutoff_energy': self.config.cutoff_energy,
            'thermal_energy': self.kT
        }
