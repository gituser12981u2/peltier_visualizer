import numpy as np
import matplotlib.pyplot as plt

from src.statistical.core.bosonic.boson_statistics \
    import BoseEinsteinConfig, BoseEinsteinStatistics
from src.statistical.core.fermionic.fermion_statistics \
    import BandGapConfig, CarrierStatisticsConfig, DielectricConfig, \
    FermiDiracStatistics
from src.statistical.core.statistical_evolution \
    import BosoniccEvolution, FermionicEvolution, ParticleEvolution
from src.statistical.thermal.debye_model import DebyeModel, DebyeModelConfig
from src.statistical.thermal.thermal_properties import ThermalProperties
from src.statistical.transport_properties import TransportProperties


def plot_results(results, transport_props):
    # Figure 1: Carrier Properties
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('Carrier Properties')

    # Carrier density
    ax1.plot(results['temperature'], results['carrier_density'], 'b-')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Carrier Density (m^-3)')
    ax1.set_yscale('log')
    ax1.set_title('Carrier Density Evolution')
    ax1.grid(True)

    # Chemical potential
    ax2.plot(results['temperature'], results['chemical_potential'], 'r-')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Chemical Potential (eV)')
    ax2.set_title('Chemical Potential Evolution')
    ax2.grid(True)

    plt.tight_layout()

    # Figure 2: Phonon Properties
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Phonon Properties')

    # Phonon population
    ax3.plot(results['temperature'], results['phonon_population'], 'g-')
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Phonon Population (m^-3)')
    ax3.set_yscale('log')
    ax3.set_title('Phonon Population Evolution')
    ax3.grid(True)

    # Thermal conductivity
    ax4.plot(results['temperature'], results['thermal_conductivity'], 'k-')
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Thermal Conductivity (W/m·K)')
    ax4.set_title('Thermal Conductivity Evolution')
    ax4.grid(True)

    plt.tight_layout()

    # Figure 3: Scattering Properties
    fig3, ax5 = plt.subplots(figsize=(8, 6))
    fig3.suptitle('Carrier Scattering Rates')

    ax5.plot(results['temperature'], results['scattering_rate'],
             'b-', label='Total')
    ax5.plot(results['temperature'], results['emission_rate'],
             'r--', label='Emission')
    ax5.plot(results['temperature'], results['absorption_rate'],
             'g--', label='Absorption')
    ax5.set_xlabel('Temperature (K)')
    ax5.set_ylabel('Scattering Rate (s⁻¹)')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()

    # Figure 4: Thermoelectric Properties
    fig4, (ax6, ax7) = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle('Thermoelectric Performance')

    # Calculate ZT for each temperature
    zt_values = []
    cop_values = []
    for T, rate in zip(results['temperature'], results['scattering_rate']):
        scattering_rates = {'total_rate': rate}
        zt = transport_props.calculate_zt(T, scattering_rates)
        zt_values.append(zt)

        # Calculate COP with 20K temperature difference
        T_hot = T + 10
        T_cold = T - 10
        cop = transport_props.calculate_cop(T_hot, T_cold, scattering_rates)
        cop_values.append(cop)

    # Plot ZT
    ax6.plot(results['temperature'], zt_values, 'b-')
    ax6.set_xlabel('Temperature (K)')
    ax6.set_ylabel('Figure of Merit ZT')
    ax6.set_title('Thermoelectric Figure of Merit')
    ax6.grid(True)

    # Plot COP
    ax7.plot(results['temperature'], cop_values, 'r-')
    ax7.set_xlabel('Temperature (K)')
    ax7.set_ylabel('COP')
    ax7.set_title('Coefficient of Performance\n(20K Temperature Difference)')
    ax7.grid(True)

    plt.tight_layout()

    return fig1, fig2, fig3, fig4


def main():
    # Configure silicon parameters for testing
    band_gap_config = BandGapConfig(
        E_g0=1.17,  # Band gap at 0K
        alpha=4.73e-4,
        beta=636.0
    )

    dielectric_config = DielectricConfig(
        epsilon_r0=11.7,  # Reference dielectric constant
        temp_coefficient=-1.5e-4,  # Temperature coefficient
        T0=300.0  # Reference temperature
    )

    carrier_config = CarrierStatisticsConfig(
        temperature=300,
        fermi_level=-0.2,  # 0.1 eV above intrinsic level for n-type
        effective_mass_electron=1.08,
        effective_mass_hole=0.81,
        band_gap=band_gap_config,
        dielectric_config=dielectric_config,
    )

    phonon_config = BoseEinsteinConfig(
        temperature=300,  # K
        cutoff_energy=0.063,  # eV for silicon
        lattice_constant=5.43e-10,  # m
        sound_velocity=8443.0,  # m/s
        chemical_potential=0.0,  # eV
        dispersion_type='debye',
        branch_degeneracy=3
    )

    debye_config = DebyeModelConfig(
        debye_temperature=645.0,  # K
        lattice_constant=5.43e-10,  # m
        atomic_mass=4.6637e-26,  # kg (Silicon atomic mass)
        elastic_constants={
            'c11': 165.7e9,  # Pa
            'c44': 79.6e9    # Pa
        },
        number_density=5e28,  # m^-3
        acoustic_deformation_potential=9.0  # eV
    )

    # Initialize statistical systems
    fd_stats = FermiDiracStatistics(carrier_config)
    be_stats = BoseEinsteinStatistics(phonon_config)
    debye_model = DebyeModel(debye_config, be_stats)
    thermal_props = ThermalProperties(debye_model)

    # Initialize evolution systems
    fermion_evolution = FermionicEvolution(fd_stats)
    boson_evolution = BosoniccEvolution(be_stats, debye_model, thermal_props)
    particle_evolution = ParticleEvolution(fermion_evolution, boson_evolution)

    transport_props = TransportProperties(fd_stats, be_stats, thermal_props)

    # Evolve temperature
    T_range = (100, 500)
    results = particle_evolution.evolve_temperature(*T_range, num_steps=50)

    fig1, fig2, fig3, fig4 = plot_results(results, transport_props)

    plt.show()

    # Print some key values
    print("\nKey values at representative temperatures:")
    for T in [100, 200, 300, 400, 500]:
        idx = np.argmin(np.abs(results['temperature'] - T))
        scattering_rates = {'total_rate': results['scattering_rate'][idx]}
        zt = transport_props.calculate_zt(T, scattering_rates)
        cop = transport_props.calculate_cop(T + 10, T - 10, scattering_rates)

        print(f"\nAt T = {T}K:")
        print(f"  Carrier density: {results['carrier_density'][idx]:.2e} m^-3")
        print(f"  Chemical potential: {
              results['chemical_potential'][idx]:.3f} eV")
        print(f"  Phonon population: {
              results['phonon_population'][idx]:.2e} m^-3")
        print(f"  Scattering rate: {results['scattering_rate'][idx]:.2e} s⁻¹")
        print(f"  Figure of Merit ZT: {zt:.3f}")
        print(f"  COP: {cop:.3f}")


if __name__ == "__main__":
    main()
