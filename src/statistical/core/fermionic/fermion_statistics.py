from dataclasses import dataclass
import numpy as np
from typing import Optional, Union
from scipy import constants as const
from scipy.special import roots_legendre


@dataclass
class BandGapConfig:
    """Configuration for temperature-dependent band gap calculation"""
    E_g0: float  # Band gap at 0K in eV
    alpha: float  # First Varshni parameter in eV/K
    beta: float  # Second Varshni paramter in K


@dataclass
class DielectricConfig:
    """Configuration for temperature-dependent dielectric constant"""
    epsilon_r0: float  # Reference dielectric constant at T0
    temp_coefficient: float  # Temperature coefficient in K^-1
    T0: float = 300.0  # Reference temperature in K


@dataclass
class CarrierStatisticsConfig:
    """Configuration parameters for carrier statistics calculations"""
    temperature: float  # Temperature in Kelvin
    fermi_level: float  # Fermi level in eV
    effective_mass_electron: float  # Effective mass of electrons m0
    effective_mass_hole: float  # Effective mass of holes m0
    # Temperature-dependent parameters
    band_gap: BandGapConfig
    dielectric_config: DielectricConfig
    degeneracy_factor: float = 2.0  # Spin degeneracy factor
    conduction_band_edge: float = 0.0  # Conduction band edge in eV
    valence_band_edge: Optional[float] = None  # Valence band edge in eV


class FermiDiracStatistics:
    """
    Class implementing Fermi-Dirac statistics for semiconductor carriers.

    This class handles calculations of carrier concentrations,
    Fermi-Dirac distributions, and related quantities for both
    degenerate and non-degenerate semiconductors.
    """

    def __init__(self, config: CarrierStatisticsConfig):
        """
        Initialize FermiDiracStatistics with material parameters.

        Args:
            config: Configuration parameters for the calculations
        """
        self.config = config

        # Set up fundamental constants
        self.kb = const.k  # Boltzmann constant
        self.kb_eV = const.k / const.e  # Boltzmann constant in eV/k
        self.h = const.h  # Planck constant
        self.hbar = const.hbar  # Reduced Planck constant
        self.m0 = const.m_e  # Electron rest mass
        self.q = const.e  # Elementary charge

        # Calculate band gap and set band edges
        self.kT = self.kb_eV * config.temperature
        band_gap = self._calculate_band_gap(config.temperature)

        # Set band edges with conduction band at 0 eV reference
        self.config.conduction_band_edge = 0
        self.config.valence_band_edge = -band_gap

    def _calculate_band_gap(self, T: float) -> float:
        """Simple band gap calculation using Varshni equation"""
        bgc = self.config.band_gap
        return bgc.E_g0 - (bgc.alpha * T**2)/(T + bgc.beta)

    def fermi_dirac(self,
                    E: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the Fermi-Dirac distribution function.

        Args:
            E: Energy level(s) in eV

        Returns:
            Fermi-Dirac occupation probability
        """
        # Handle potential overflow in exp
        eta = (E - self.config.fermi_level) / self.kT
        if eta > 100:
            return 0.0
        elif eta < -100:
            return 1.0
        return 1 / (1 + np.exp(eta))

    def density_of_states_3D(self, E: float, effective_mass: float) -> float:
        """
        Calculate the 3D density of states.

        Args:
            E: Energy in eV
            effective_mass: Effective mass in units of m0

        Returns:
            Density of states in states/(eV*m^3)
        """
        m_eff = effective_mass * self.m0

        # Calculate prefactor (m^-3 J^-1/2)
        prefactor = (4 * np.pi * (2 * m_eff)**(3/2)) / \
            (self.h**3) * (self.q)**(3/2)

        return prefactor * np.sqrt(abs(E)) if E > 0 else 0.0

    def calculate_carrier_density(self,
                                  carrier_type: str = 'electron') -> float:
        """
        Calculate carrier density by integrating over energy states.

        The carrier density is calculated by integrating the product of the
        density of states and the Fermi-Dirac distribution over the
        appropriate energy range:

        n = ∫ D(E) f(E) dE

        For electrons, integrate from the conduction band edge upward.
        For holes, integrate from the valence band edge downward.

        Args:
            carrier_type: Either electron or hole
            temp_override: Optional temperature override for calculation

        Returns:
            Carrier density in carries/m^3
        """
        # Store original temperature if using override
        is_electron = carrier_type.lower() == 'electron'
        band_edge = (self.config.conduction_band_edge if is_electron
                     else self.config.valence_band_edge)
        effective_mass = (self.config.effective_mass_electron
                          if is_electron
                          else self.config.effective_mass_hole)

        # Calculate DOS at band edge for normalization
        Nc = self.calculate_effective_dos(carrier_type)

        def integrand(E_relative: float) -> float:
            """
            Calculate the integrand D(E)f(E) at a given energy.

            Args:
                E_relative: Energy relative to band edge in eV

            Returns:
                Value of D(E)f(E) at given energy
            """
            # Get absolute energy
            E = band_edge + (E_relative if is_electron else -E_relative)

            dos = self.density_of_states_3D(E_relative, effective_mass)

            # Calculate occupation probability
            occupation = self.fermi_dirac(E)

            # For holes, system interested in unoccupied states
            if not is_electron:
                occupation = 1.0 - occupation

            return dos * occupation / Nc  # Normalize by DOS

        # Improve integration accuracy
        E_max = max(
            50 * self.kT, abs(self.config.fermi_level - band_edge)
            + 20 * self.kT)

        # Split integration into regions
        ranges = [
            # More points near band edges where integrand varies rapidly
            (0, 5 * self.kT),  # Near band edge
            (5 * self.kT, 20 * self.kT),  # Intermediate energy range
            (20 * self.kT, E_max)  # Extended energy range
        ]

        # Use more points near band edge
        points_per_range = [100, 80, 60]

        total_density = 0.0
        for (a, b), n_points in zip(ranges, points_per_range):
            # Get quadrature points and weights
            x, w = roots_legendre(n_points)

            # Transform integration range from [-1, 1] to [a, b]
            for xi, wi, in zip(x, w):
                E = ((b-a)/2) * xi + (a+b)/2  # Transform to actual energy
                total_density += wi * integrand(E) * (b-a)/2

        return max(0.0, total_density * Nc)  # Denormalize

    def find_fermi_level(self, target_density: float,
                         carrier_type: str = 'electron') -> float:
        """
        Find the Fermi level that gives a target carrier density.

        Args:
            target_density: Target carrier density in carriers/m^3
            carrier_type: Either electron or hole

        Returns:
            Fermi level in eV
        """
        def objective(E_f: float) -> float:
            old_fermi = self.config.fermi_level
            self.config.fermi_level = E_f
            density = self.calculate_carrier_density(carrier_type)
            self.config.fermi_level = old_fermi
            return density - target_density

        # Initial guess for Fermi level
        if carrier_type == 'electron':
            E_guess = self.config.conduction_band_edge - self.kT * \
                np.log(target_density /
                       self.calculate_effective_dos('electron'))
        else:
            E_guess = self.config.valence_band_edge + self.kT * \
                np.log(target_density / self.calculate_effective_dos('hole'))

        from scipy.optimize import root_scalar
        result = root_scalar(
            objective,
            x0=E_guess,
            x1=E_guess + 0.1,
            method='secant'
        )

        return result.root

    def calculate_effective_dos(self, carrier_type: str = 'electron') -> float:
        """
        Calculate the effective density of states.

        Args:
            carrier_type: Either 'electron' or 'hole'

        Returns:
            Effective density of states in states/m³
        """
        effective_mass = (self.config.effective_mass_electron if
                          carrier_type == 'electron' else
                          self.config.effective_mass_hole)

        return 2 * (2 * np.pi * effective_mass * self.m0 * self.kb *
                    self.config.temperature / (self.h ** 2)) ** (3/2)

    def degeneracy_check(self, carrier_type: str = 'electron') -> float:
        """
        Check the degeneracy level of the semiconductor.

        Returns a dimensionless parameter η = (Ef - Ec)/kT for electrons
        or η = (Ev - Ef)/kT for holes. |η| >> 1 indicates strong degeneracy.

        Args:
            carrier_type: Either electron or hole

        Returns:
            Degeneracy parameter η
        """
        if carrier_type == 'electron':
            return (self.config.fermi_level
                    - self.config.conduction_band_edge) / self.kT
        else:
            return (self.config.valence_band_edge
                    - self.config.fermi_level) / self.kT

    def calculate_chemical_potential(self, T: float) -> float:
        """
        Calculate the chemical potential variation with temperature.

        Args:
            T: Temperature (K)

        Returns:
            Chemical potential in eV
        """
        # Store original temperature
        T_original = self.config.temperature

        # Set new temperature
        self.config.temperature = T
        self.kT = self.kb_eV * T

        # Get current carrier density
        n = self.calculate_carrier_density('electron')
        p = self.calculate_carrier_density('hole')

        # Find new Fermi level that maintains charge neutrality
        def charge_neutrality(E_f):
            self.config.fermi_level = E_f
            n_new = self.calculate_carrier_density('electron')
            p_new = self.calculate_carrier_density('hole')
            return n_new - p_new - (n - p)

        from scipy.optimize import root_scalar
        result = root_scalar(
            charge_neutrality,
            x0=self.config.fermi_level,
            x1=self.config.fermi_level + 0.1,
            method='secant'
        )

        # Restore original temperature
        self.config.temperature = T_original
        self.kT = self.kb_eV * T_original

        return result.root

    def calculate_dielectric_constant(self, T: float) -> float:
        """
        Calculate temperature-dependent dielectric constant.

        Args:
            T: Temperature in Kelvin

        Returns:
            Relative dielectric constant at temperature T
        """
        dc = self.config.dielectric_config
        return dc.epsilon_r0 * (1 + dc.temp_coefficient * (T - dc.T0))

    def update_temperature_dependent_parameters(self, T: float):
        """
        Update all temperature-dependent parameters.

        Args:
            T: Temperature in Kelvin
        """
        # Update temperature and thermal energy
        self.temperature = T
        self.kT = self.kb_eV * T

        # Update dielectric constant
        self.dielectric_constant = self.calculate_dielectric_constant(T)

    def update_band_structure(self, T: float):
        """Update band structure separately from temperature parameters"""
        # Calculate new band edges
        Ec, Ev = self.band_structure.calculate_band_edges(T)
        self.config.conduction_band_edge = Ec
        self.config.valence_band_edge = Ev

        # Calculate intrinsic level if needed
        self.intrinsic_fermi_level = (Ec + Ev) / 2
        if not hasattr(self.config, 'fermi_level'):
            self.config.fermi_level = self.intrinsic_fermi_level

    def carrier_summary(self) -> dict:
        """
        Generate a summary of carrier statistics.

        Returns:
            Dictionary containing key carrier statistics
        """
        n = self.calculate_carrier_density('electron')
        p = self.calculate_carrier_density('hole')

        return {
            'electron_density': n,
            'hole_density': p,
            'electron_degeneracy': self.degeneracy_check('electron'),
            'hole_degeneracy': self.degeneracy_check('hole'),
            'effective_dos_electron': self.calculate_effective_dos('electron'),
            'effective_dos_holes': self.calculate_effective_dos('hole'),
            'intrinsic_carrier_density': np.sqrt(n * p),
            'fermi_level': self.config.fermi_level,
            'temperature': self.config.temperature
        }
