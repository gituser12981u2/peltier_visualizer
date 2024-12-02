from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional


class ScatteringModel(ABC):
    """Abstract base class for scattering models"""

    @abstractmethod
    def calculate_scattering_rate(self, energy: float,
                                  temperature: float) -> float:
        """Calculate scattering rate for given energy and temperature"""
        pass

    @abstractmethod
    def update(self, time: float, temperature: float):
        """Update any time-dependent properties"""
        pass


class FermiDiracScattering(ScatteringModel):
    """Electron-electron scattering under Fermi-Dirac statistics"""

    def __init__(self, fd_stats):
        """
        Initialize scattering model.

        Args:
            fd_stats: FermiDiracStatistics instance
        """
        self.stats = fd_stats
        self._cached_rates: Dict[tuple, float] = {}
        self.epsilon = self.stats.config.dielectric_constant * \
            8.854e-12  # Silicon dielectric constant (F/m)
        self.epsilon_eV = self.epsilon / (self.stats.q**2)  # epsilon in eV*m

    def calculate_scattering_rate(self, energy: float,
                                  temperature: float) -> float:
        """
        Calculate e-e scattering rate using Fermi's golden rule

        Args:
            energy: Electron energy (eV)
            temperature: Temperature (K)

        Returns:
            Scattering rate (s^-1)
        """
        cache_key = (round(energy, 6), round(temperature, 2))
        if cache_key in self._cached_rates:
            return self._cached_rates[cache_key]

        # Convert input energy to Joules
        kT = self.stats.kb * temperature
        m_eff = self.stats.config.effective_mass_electron * self.stats.m0

        # Calculate wavevector
        E_joules = energy * self.stats.q
        k = np.sqrt(2 * m_eff * E_joules) / self.stats.hbar

        # Calculate screening length (Thomas-Fermi)
        n = self.stats.calculate_carrier_density('electron')
        screening_length = np.sqrt(
            self.epsilon * kT / (n * self.stats.q**2))
        # maybe use screen_length = debye length * np.sqrt((2/3) E_f/kT)

        # Initialize rate with Fermi's golden rule prefactor
        rate = 2 * np.pi / self.stats.hbar

        # Small broadening for energy conservation (in eV)
        gamma = 1e-3 * kT

        # Angular integration for momentum conservation
        theta_points = 50
        theta = np.linspace(0, np.pi, theta_points)
        d_theta = np.pi/theta_points

        angular_sum = 0
        for th in theta:
            q = 2 * k * np.sin(th/2)  # Momentum transfer
            if q == 0:
                continue

            # TODO: make screened potential dynamic
            # Screened Coulomb potential
            V_q = self.stats.q**2 / \
                (self.epsilon * (q**2 + screening_length**-2))

            # Collision energy transfer (in eV)
            dE = (self.stats.hbar * q)**2 / (2 * m_eff * self.stats.q)

            # Phase space factors
            E1 = energy
            E2 = energy - dE
            E3 = energy + dE
            E4 = energy

            E_transfer = E3 - E1
            energy_conservation = np.exp(-(dE - E_transfer)
                                         ** 2 / (2 * gamma**2))

            f1 = self.stats.fermi_dirac(E1)
            f2 = self.stats.fermi_dirac(E2)
            f3 = self.stats.fermi_dirac(E3)
            f4 = self.stats.fermi_dirac(E4)

            PS = f1 * f2 * (1 - f3) * (1 - f4)

            # Matrix element
            M_sq = V_q**2 * (1 + np.cos(th)**2)

            # Add contribution from angle in spherical coordinates
            angular_sum += M_sq * PS * \
                energy_conservation * np.sin(th) * d_theta

        # Final normalization
        rate *= angular_sum

        self._cached_rates[cache_key] = rate
        return rate

    def update(self, time: float, temperature: float):
        """Clear cache if conditions change significantly"""
        if abs(temperature - self.stats.config.temperature) > 1:
            self._cached_rates.clear()
            # Update epsilon if it depends on temperature
            self.epsilon = self.stats.config.dielectric_constant * 8.854e-12
        # ?? Maybe add dynamic screening updates?

    def get_energy_transfer_rate(self, energy: float,
                                 temperature: float) -> float:
        """
        Calculate energy transfer rate for Joule heating

        Args:
            energy: Initial electron energy (eV)
            temperature: Temperature (K)

        Returns:
            Energy transfer rate (eV/s)
        """
        scattering_rate = self.calculate_scattering_rate(energy, temperature)
        # Average energy exchanged
        avg_energy_transfer = self.stats.kT
        return scattering_rate * avg_energy_transfer


# TODO: make work
class BoseEinsteinScattering(ScatteringModel):
    """Placeholder for phonon-phonon and electron-phonon scattering"""

    def __init__(self):
        self.phonon_populations = {}  # To be implemented

    def calculate_scattering_rate(self, energy: float,
                                  temperature: float) -> float:
        # Placeholder - implement actual phonon scattering
        return 0.0

    def update(self, time: float, temperature: float):
        # Update phonon populations
        pass


class CombinedScatteringModel:
    """Combines different scattering mechanisms"""

    def __init__(self, fd_scattering: FermiDiracScattering,
                 be_scattering: Optional[BoseEinsteinScattering] = None):
        self.fd_scattering = fd_scattering
        self.be_scattering = be_scattering

    def get_total_scattering_rate(self, energy: float,
                                  temperature: float) -> float:
        """Calculate total scattering rate using Matthiessen's rule"""
        rate = self.fd_scattering.calculate_scattering_rate(
            energy, temperature)

        if self.be_scattering:
            rate += self.be_scattering.calculate_scattering_rate(
                energy, temperature)

        return rate

    def update(self, time: float, temperature: float):
        """Update all scattering models"""
        self.fd_scattering.update(time, temperature)
        if self.be_scattering:
            self.be_scattering.update(time, temperature)
