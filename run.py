import math
import sys

import argparse
import csv
import numpy as np
import os
from PySide6.QtWidgets import QApplication
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
import simulation as sim
import visualize as vis
from analysis import plot_displacement_vs_time_line


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Particle Simulation Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument(
        '--total_time', type=int, default=2000,
        help='Maximum simulation time (seconds)'
    )
    sim_group.add_argument(
        '--n_particles', type=int, default=1,
        help='Number of particles to simulate'
    )
    sim_group.add_argument(
        '--p_driv', type=float, default=0.03,
        help='Probability of driven motion (0.0-1.0)'
    )
    sim_group.add_argument(
        '--trap_dist', type=float, default=sim.TRAP_DIST,
        help=f'Distance between traps (meters) (default: {sim.TRAP_DIST})'
    )
    sim_group.add_argument(
        '--trap_std', type=float, default=sim.TRAP_STD,
        help=f'Standard deviation of trap distance (meters) (default: {sim.TRAP_STD})'
    )
    sim_group.add_argument(
        '--time_between', type=float, default=sim.TIME_BETWEEN_STATES,
        help=f'Average time between state changes (seconds) (default: {sim.TIME_BETWEEN_STATES})'
    )
    sim_group.add_argument(
        '--dt', type=float, default=0.001,
        help='Length of time step for simulation (seconds)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--dirname', type=str, default='sim',
        help='Directory name for output files'
    )
    output_group.add_argument(
        '--record_frames', action='store_true',
        help='Automatically start recording frames on startup'
    )

    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument(
        '--width', type=int, default=600,
        help='Canvas width in pixels'
    )
    display_group.add_argument(
        '--height', type=int, default=600,
        help='Canvas height in pixels'
    )

    # Special modes
    mode_group = parser.add_argument_group('Special Modes')
    mode_group.add_argument(
        '--headless', action='store_true',
        help='Run simulation without GUI (exports data only)', default=False
    )
    mode_group.add_argument(
        '--compute_flux', action='store_true',
        help='Compute and plot flux distribution for particles', default=False
    )

    # Help and examples
    parser.epilog = """
Examples:
  %(prog)s                                    # Run with default settings
  %(prog)s --n_particles 5 --total_time 5000 # Simulate 5 particles for 5000 steps
  %(prog)s --p_driv 0.1 --dirname results     # Higher drive probability, custom output
  %(prog)s --width 800 --height 800          # Larger display window
  %(prog)s --headless --dirname run          # Run without GUI
    """

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments"""
    errors = []

    # Validate ranges
    if args.total_time <= 0:
        errors.append("total_time must be positive")
    if args.n_particles <= 0:
        errors.append("n_particles must be positive")
    if not 0.0 <= args.p_driv <= 1.0:
        errors.append("p_driv must be between 0.0 and 1.0")
    if args.dt <= 0:
        errors.append("dt must be positive")
    if args.width <= 0 or args.height <= 0:
        errors.append("width and height must be positive")

    # Validate optional parameters
    if args.trap_dist is not None and args.trap_dist <= 0:
        errors.append("trap_dist must be positive")
    if args.time_between is not None and args.time_between <= 0:
        errors.append("time_between must be positive")

    # Display errors
    if errors:
        print("Error: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True

def run_headless_simulation(config: sim.SimulationConfig) -> bool:
    """Run the simulation without a GUI and export results to CSV."""
    print(f"Running headless simulation with {config.n_particles} particles…")
    print(f"Output directory: {config.dirname}")

    try:
        os.makedirs(config.dirname, exist_ok=True)

        coords, exit_times = [], []
        max_n_steps = n_exited = 0

        for i in tqdm(range(config.n_particles),
                      desc="Particles",
                      unit="particle"):
            x_coords, y_coords, exit_time, *_ = sim.move(
                config,
                theta=i * 2 * np.pi / config.n_particles
            )

            max_n_steps = max(len(x_coords), max_n_steps)
            if exit_time != -1:
                n_exited += 1
                exit_times.append(exit_time)

            coords.append([x_coords, y_coords])

        # Export to CSV
        csv_filename = os.path.join(config.dirname, "coords.csv")
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "particle", "x", "y"])
            for p_idx, (xs, ys) in enumerate(coords):
                for frame, (x, y) in enumerate(zip(xs, ys)):
                    writer.writerow([frame, p_idx, x, y])

        # Summary
        print("\nSimulation complete!")
        print("======================\n")
        print(f"Particles simulated : {len(coords)}")
        print(f"Particles exited    : {n_exited}")
        print(f"Maximum steps       : {max_n_steps}")
        if exit_times:
            print(f"Average exit time   : {np.mean(exit_times):.2e} s")
        print(f"\nData saved -> {csv_filename}")

        return True

    except Exception as e:
        print(f"Error running simulation: {e}")
        return False

def main():
    """Main application entry point"""
    # Parse and validate arguments
    args = parse_arguments()

    if not validate_arguments(args):
        sys.exit(1)

    # Create configuration
    config = sim.SimulationConfig.from_args(args)

    # Print configuration
    print(f"Particle Simulation Visualizer")
    print(f"==============================")
    print(f"Configuration:")
    print(f"  Number of particles: {config.n_particles}")
    print(f"  Total time: {config.total_time} s")
    print(f"  Probability of driven motion: {config.p_driv}")
    print(f"  Output directory: {config.dirname}")
    if args.headless:
        print(f"  Mode: Headless (no GUI)")
    else:
        print(f"  Display: {config.width}x{config.height}")
    print()

    # Run headless simulation if requested
    if args.headless:
        success = run_headless_simulation(config)
        if args.compute_flux:
            plot_flux_distribution(config)
        sys.exit(0 if success else 1)

    # Run GUI application
    app = QApplication(sys.argv)

    # simulation = vis.SimulationVis(config)
    # simulation.show()
    # simulation.run_simulation()

    if args.compute_flux:
        plot_flux_distribution(config)

    sys.exit(app.exec())

def generate_displacement_time_driven_graph(n_particles = 50, dt = 0.001, total_time = 600):
    import numpy as np
    import matplotlib.pyplot as plt
    import simulation as sim

    do_analysis = True

    if do_analysis:

        # Create chart
        plt.figure()

        # Calculate and plot lines for each p_driv value on chart
        plot_displacement_vs_time_line(
            total_time  = total_time,
            n_particles = n_particles,
            p_driv_vals = [1,.50,.24,.12,.06,.03,0],
            colors = ['#2ba03b', '#e77cc2', '#ecb01f', '#926aba', 'red', '#0079b1', '#f98436']
        )

        # Add labels, title, legend to chart
        plt.xlabel("time (s)")
        plt.ylabel("Displacement (µm)")
        plt.title(f"Displacement vs. Time for Varying Driven Motion Amounts")
        plt.legend()
        plt.xlim([0, math.ceil(total_time / 1000) * 1000]) # Round total time up to the nearest thousand
        plt.ylim([0, 50])

        # Show chart
        plt.show()


def plot_flux_distribution(config,
                           snapshot_interval: float = 0.01,
                           window_duration: int = 4,
                           diffusive_threshold: float = 2.5e-9):
    """
    Plot the distribution of 4‑second flux values in **micrometers** for every particle.

    Parameters
    ----------
    config : SimulationConfig
        Configuration object with simulation parameters (must include n_particles).
    snapshot_interval : float, optional
        Time step (s) between snapshots used when calling `compute_flux_4s`.
    window_duration : int, optional
        Window (s) over which diffusive flux is computed.
    diffusive_threshold : float, optional
        Threshold (m) used to decide whether a step counts toward diffusive flux.

    Notes
    -----
    * All flux values returned by the simulation are assumed to be in **meters**.
      They are converted to **micrometers** (× 1 000 000) before plotting.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from tqdm import tqdm

    import simulation as sim
    from analysis import compute_flux_4s

    # Accumulate flux values from every particle
    all_flux_diffusive = []
    all_flux_driven = []

    for i in tqdm(range(config.n_particles), desc="Calculating 4‑s flux", unit="particle"):
        theta = 2 * np.pi * i / config.n_particles
        sim_output = sim.move(config, theta=theta)

        flux_diffusive = compute_flux_4s(sim_output,
                                         config,
                                         snapshot_interval,
                                         window_duration,
                                         diffusive_threshold)
        flux_driven = sim_output.distance_driven

        all_flux_diffusive.extend(flux_diffusive)
        all_flux_driven.extend(flux_driven)

    # ---------- Convert from meters → micrometers ----------
    all_flux_diffusive_um = np.asarray(all_flux_diffusive) * 1e6  # μm
    all_flux_driven_um = np.asarray(all_flux_driven) * 1e6        # μm

    # ---------- Histogram settings ----------
    weights_diffusive = np.ones_like(all_flux_diffusive_um) / len(all_flux_diffusive_um) * 100
    weights_driven = np.ones_like(all_flux_driven_um) / len(all_flux_driven_um) * 100

    # Remove zeros for log‑spaced bins
    flux_diffusive_nz = all_flux_diffusive_um[all_flux_diffusive_um > 0]
    flux_driven_nz = all_flux_driven_um[all_flux_driven_um > 0]
    all_nz = np.concatenate([flux_diffusive_nz, flux_driven_nz])

    num_bins = 60
    bins = np.logspace(np.log10(all_nz.min()), np.log10(all_nz.max()), num_bins)

    # -- Print summary statistics --
    print(f"Diffusive flux values (μm): mean {np.mean(all_flux_diffusive_um):.4e}, std {np.std(all_flux_diffusive_um):.4e}")
    print(f"Driven flux values (μm): mean {np.mean(all_flux_driven_um):.4e}, std {np.std(all_flux_driven_um):.4e}")

    # ---------- Plot ----------
    plt.figure()

    plt.hist(all_flux_diffusive_um, bins=bins, weights=weights_diffusive,
             color='blue', alpha=0.7, edgecolor='black', label='Diffusive Flux')
    plt.hist(all_flux_driven_um, bins=bins, weights=weights_driven,
             color='red',  alpha=0.7, edgecolor='black', label='Driven Flux')

    plt.xscale('log')
    plt.xlabel(r"Flux ($\mu$m)")
    plt.ylabel("Percentage")
    plt.title("Distribution of 4‑Second Flux Values")
    plt.gca().yaxis.set_major_formatter(PercentFormatter())  # y‑axis in %
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
    # generate_displacement_time_driven_graph(n_particles=20, total_time=600)
    # generate_displacement_time_driven_graph(n_particles=20, total_time=3600)



