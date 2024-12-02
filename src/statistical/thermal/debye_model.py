from dataclasses import dataclass
from typing import Optional, Union, Dict
import numpy as np
from scipy import constants as const
from src.statistical.core.bosonic.boson_statistics import \
    BoseEinsteinStatistics


@dataclass
class DebyeModelConfig:
    """Configuration parameters for the Debye model"""
    debye_temperature: float  # Kelvin
    lattice_constant: float  # meters
    atomic_mass: float  # kg
    elastic_constants: Dict[str, float]  # Elastic constants in N/m^2
    number_density: float  # Number of atoms per unit volume in m^-3
    acoustic_deformation_potential: float  # Deformation potential in eV


class DebyeModel:
    """
    Implementation of the Debye model for phonon calculations
    in semiconductors.

    The class handles phonon dispersion relations, density of states,
    and electron-phonon coupling calculations.
    """

    def __init__(self, config: DebyeModelConfig,
                 be_stats: Optional[BoseEinsteinStatistics] = None):
        """
        Initialize Debye model with material parameters.

        Args:
            config: Configuration parameters for the Debye model
            be_stats: Optional BoseEinstein Statistics instance for
            population calculations
        """
        self.config = config
        self.be_stats = be_stats

        # Fundamental constants
        self.kb = const.k
        self.hbar = const.hbar
        self.q = const.e

        # Derived quantities
        self.debye_frequency = self.kb * self.config.debye_temperature \
            / self.hbar
        self.debye_wavevector = self.calculate_debye_wavevector()

    def calculate_debye_wavevector(self) -> float:
        """Calculate the Debye wavevector from number density"""
        return (6 * np.pi**2 * self.config.number_density)**(1/3)

    def dispersion_acoustic(self, q: Union[float, np.ndarray],
                            branch: str = 'LA') -> Union[float, np.ndarray]:
        """
        Calculate acoustic phonon dispersion relation.

        Args:
            q: Wavevector magnitude in m^-1
            branch: 'LA' for longitudinal or 'TA' for transverse acoustic

        Returns:
            Phonon frequency in rad/s
        """
        # Get relevant elastic constant
        c_ij = (self.config.elastic_constants['c11'] if branch == 'LA'
                else self.config.elastic_constants['c44'])

        # Calculate sound velocity
        v_sound = np.sqrt(c_ij / self.config.atomic_mass)

        # Apply sine correction for large q
        q_max = np.pi / self.config.lattice_constant

        # Handle q->0 limit using Taylor expansion
        x = np.pi * q / (2 * q_max)
        if isinstance(q, np.ndarray):
            small_q = x < 1e-10
            result = np.zeros_like(q, dtype=float)
            result[small_q] = v_sound * q[small_q]
            result[~small_q] = v_sound * q[~small_q] * \
                np.sin(x[~small_q]) / x[~small_q]
            return result
        else:
            if x < 1e-10:
                return v_sound * q
            return v_sound * q * np.sin(x) / x

    def density_of_states(self, omega: float) -> float:
        """
        Calculate phonon density of states using the Debye model.

        Args:
            omega: Phonon frequency in rad/s
        """
        if omega > self.debye_frequency or omega < 0:
            return 0.0

        return (9 * self.config.number_density * omega**2) \
            / (self.debye_frequency**3)
