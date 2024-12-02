import numpy as np
from scipy import constants as const
from scipy.integrate import quad
from src.statistical.core.bosonic.boson_statistics \
    import BoseEinsteinStatistics
from src.statistical.core.fermionic.fermion_statistics \
    import FermiDiracStatistics
from src.statistical.thermal.debye_model import DebyeModel


class ParticleInteractions:
    """
    Handles interactions between electrons and phonons in semiconductors.
    This class combines Fermi-Dirac and Bose-Einstein statistics with
    the Debye model to calculate scattering rates and coupling effects.
    """

    def __init__(self, fd_stats: FermiDiracStatistics,
                 be_stats: BoseEinsteinStatistics, debye_model: DebyeModel):
        self.fd_stats = fd_stats
        self.be_stats = be_stats
        self.debye_model = debye_model

        # Physical constants
        self.kb = const.k
        self.hbar = const.hbar
        self.q = const.e

    def electron_phonon_coupling(self, q: float, E: float,
                                 deformation_potential: float) -> float:
        """
        Calculate electron-phonon coupling matrix element with screening.

        Args:
            q: Phonon wavevector in m^-1
            E: Electron energy in eV
            deformation_potential: Acoustic deformation potential in eV

        Returns:
            Coupling matrix element squared in ev^2
        """
        # Acoustic deformation potential approximation
        rho = self.debye_model.config.atomic_mass * \
            self.debye_model.config.number_density
        V = 1.0  # Normalization volume

        omega_q = self.debye_model.dispersion_acoustic(q)

        # Thomas-Fermi screening
        qTF = 1.0e8  # Typical screening wavevector
        screening = 1.0 / (1.0 + (q/qTF)**2)

        # Add cutoff for large q
        q_cutoff = self.debye_model.debye_wavevector / 2
        suppression = np.exp(-(q - q_cutoff)**2 / (0.1 * q_cutoff)**2) \
            if q > q_cutoff else 1.0

        return (self.hbar * deformation_potential**2 * q) \
            / (2 * rho * omega_q * V) * screening * suppression

    def calculate_scattering_rate(self,
                                  electron_energy: float,
                                  T: float,
                                  deformation_potential: float) -> float:
        """
        Calculate various scattering rates.

        Args:
            electron_energy: Initial electron energy in eV
            T: Temperature in Kelvin
            deformation_potential: Acoustic deformation potential in eV

        Returns:
            Scattering rate in s^-1
        """
        def integrand(q: float, emission: bool = True) -> float:
            """Integration kernel for q->0"""
            if q < 1e-20:  # Small cutoff for numerical stability
                return 0.0

            # Calculate energy transfer
            omega_q = self.debye_model.dispersion_acoustic(q)
            E_phonon = self.hbar * omega_q / self.q  # Convert to eV

            # Matrix element
            try:
                M_sq = self.electron_phonon_coupling(
                    q, electron_energy, deformation_potential)
            except ZeroDivisionError:
                return 0.0

            # Phase space factors including BE distributions
            n_q = self.be_stats.bose_einstein(E_phonon)

            # Separate emission and absorption processes
            occupation_factor = n_q + 1 if emission else n_q

            # Add emission and absorption contributions
            return 2 * np.pi / self.hbar * M_sq * occupation_factor * q**2

        # Split integration range
        q_mid = self.debye_model.debye_wavevector / 2

        # Integrate emission and absorption separately
        emission_rate = sum(quad(integrand, a, b, args=(True,),
                                 limit=200, epsabs=1e-30, epsrel=1e-10)[0]
                            for a, b in [(1e-20, q_mid),
                                         (q_mid,
                                          self.debye_model.debye_wavevector)
                                         ])

        absorption_rate = sum(quad(integrand, a, b, args=(False,),
                                   limit=200, epsabs=1e-30, epsrel=1e-10)[0]
                              for a, b in [(1e-20, q_mid),
                                           (q_mid,
                                            self.debye_model.debye_wavevector)
                                           ])

        return {
            'emission_rate': emission_rate,
            'absorption_rate': absorption_rate,
            'total_rate': emission_rate + absorption_rate
        }
