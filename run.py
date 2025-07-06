import sys

import argparse
import csv
import numpy as np
import os
from PySide6.QtWidgets import QApplication

import simulation as sim
import visualize as vis


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
    """Run simulation without GUI and export data"""
    print(f"Running headless simulation with {config.n_particles} particles...")
    print(f"Output directory: {config.dirname}")
    
    try:
        # Ensure output directory exists
        os.makedirs(config.dirname, exist_ok=True)
        
        # Run simulation
        coords = []
        max_n_steps = 0
        n_exited = 0
        exit_times = []
        
        for i in range(config.n_particles):
            print(f"Simulating particle {i+1}/{config.n_particles}...")
            
            x_coords, y_coords, throw_out, exit_time, _, _, _, _ = sim.move(
                config.total_time,
                config.p_driv,
                config.trap_dist,
                config.trap_std,
                config.time_between,
                2*np.pi*i/config.n_particles,
                config.dt,
            )
            
            max_n_steps = max(len(x_coords), max_n_steps)
            
            if throw_out:
                print(f"  Particle {i+1} discarded (entered nucleus)")
                continue
            
            if exit_time != -1:
                n_exited += 1
                exit_times.append(exit_time)
            
            coords.append([x_coords, y_coords])
        
        # Export results
        csv_filename = os.path.join(config.dirname, 'coords.csv')
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'particle', 'x', 'y'])
            
            for i, coord in enumerate(coords):
                for j in range(len(coord[0])):
                    writer.writerow([j, i, coord[0][j], coord[1][j]])
        
        # Write summary
        print(f"\nSimulation complete!")
        print(f"==================\n\n")
        print(f"Results:\n")
        print(f"  Particles completed: {len(coords)}\n")
        print(f"  Particles exited: {n_exited}\n")
        print(f"  Maximum steps: {max_n_steps}\n")
        if exit_times:
            print(f"  Average exit time: {np.mean(exit_times):.2e} s\n")
        
        print(f"Results saved to: {config.dirname}/")
        print(f"  - {csv_filename}")
        
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
        sys.exit(0 if success else 1)
    
    # Run GUI application
    app = QApplication(sys.argv)
    
    simulation = vis.Simulation(config)
    simulation.show()
    simulation.run_simulation()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()