import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


class ScatteringVisualizer:
    def __init__(self, fd_stats, scattering_model, box_size=100e-9):
        self.stats = fd_stats
        self.scattering = scattering_model
        self.box_size = box_size

        # Calculate number of electrons from carrier density
        self.n = self.stats.calculate_carrier_density('electron')
        # Reduce for visualization
        self.num_particles = int(self.n * box_size**3 / 100)

        # Initialize particle positions and velocities
        self.positions = box_size * np.random.random((self.num_particles, 2))

        # Initialize velocities based on Maxwell-Boltzmann distribution
        kT = self.stats.kb * self.stats.config.temperature
        v_th = np.sqrt(2 * kT / (self.stats.m0 *
                       self.stats.config.effective_mass_electron))
        self.velocities = np.random.normal(
            0, v_th/np.sqrt(2), (self.num_particles, 2))

        # Track collision events and energy
        self.collision_events = []
        self.total_energy = []
        self.time = 0

    def update(self, frame):
        dt = 1e-14  # 10 fs time step
        self.time += dt

        # Update positions
        self.positions += self.velocities * dt

        # Periodic boundary conditions
        self.positions %= self.box_size

        # Check for collisions
        for i in range(self.num_particles):
            for j in range(i+1, self.num_particles):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < 1e-9:  # 1nm interaction radius
                    # Calculate scattering rate for these electrons
                    E = 0.5 * self.stats.m0 * \
                        np.linalg.norm(self.velocities[i])**2
                    rate = self.scattering.calculate_scattering_rate(
                        E, self.stats.config.temperature)

                    # Probabilistic scattering
                    if np.random.random() < rate * dt:
                        # Elastic collision
                        v1, v2 = self.velocities[i], self.velocities[j]
                        self.velocities[i], self.velocities[j] = v2, v1
                        self.collision_events.append(
                            (self.time, self.positions[i], self.positions[j]))

        # Calculate system energy
        E_total = np.sum(0.5 * self.stats.m0 *
                         np.sum(self.velocities**2, axis=1))
        self.total_energy.append((self.time, E_total))

        return self.positions, self.velocities, self.collision_events, self.total_energy

    def create_animation(self, num_frames=200):
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)

        # Particle motion plot
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter([], [], c='b', alpha=0.6)
        ax1.set_xlim(0, self.box_size)
        ax1.set_ylim(0, self.box_size)
        ax1.set_title('Electron Motion')

        # Energy plot
        ax2 = fig.add_subplot(gs[0, 1])
        energy_line, = ax2.plot([], [])
        ax2.set_title('Total energy vs Time')

        # Scattering rate plot
        ax3 = fig.add_subplot(gs[1, 0])
        rate_line, = ax3.plot([], [])
        ax3.set_title('Scattering Rate vs Energy')

        # Statistics text
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        stats_text = ax4.text(0.1, 0.9, '', transform=ax4.transAxes)

        def init():
            return scatter, energy_line, rate_line, stats_text

        def animate(frame):
            positions, velocities, collisions, energy = self.update(frame)

            # Update particle positions
            scatter.set_offsets(positions)

            # Update energy plot
            times, energies = zip(*self.total_energy)
            energy_line.set_data(times, energies)
            ax2.relim()
            ax2.autoscale_view()

            # Update scattering rate plot
            energies = np.linspace(0, 1, 100)
            rates = [self.scattering.calculate_scattering_rate(
                E, self.stats.config.temperature)
                for E in energies]
            rate_line.set_data(energies, rates)
            ax3.relim()
            ax3.autoscale_view()

            # Update statistics text
            stats = (
                f"Temperature: {self.stats.config.temperature:.1f} K\n"
                f"Carrier Density: {self.n:.2e} m^-3\n"
                f"Band Gap: {self.stats.config.band_gap:.2f} eV\n"
                f"Fermi Level: {self.stats.config.fermi_level:.2f} eV\n"
                f"Collision Count: {len(self.collision_events)}"
            )
            stats_text.set_text(stats)

            return scatter, energy_line, rate_line, stats_text

        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=num_frames, interval=50, blit=True)

        plt.tight_layout()
        return fig, anim
