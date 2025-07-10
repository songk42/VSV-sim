from typing import NamedTuple

import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib.cbook import print_cycles

import simulation as sim
import tqdm

def writefile(x, y, i, p_driv):
    if not os.path.exists("coord-test"):
        os.makedirs("coord-test")
    f = open("coord-test/coords-{}-{}".format(p_driv, i), 'w')
    for j in range(len(x)):
        f.write("{},{}\n".format(x[j], y[j]))
    f.close()


class AnalysisOutput(NamedTuple):
    msd: np.ndarray  # Mean squared displacement
    exit_times: np.ndarray  # Times at which particles exited the cell
    max_n_steps: int  # Maximum number of steps taken by any particle
    flux_trap: np.ndarray  # Flux for hopping motion
    flux_driven: np.ndarray  # Flux for driven motion
    distance_trap: np.ndarray  # Distance traveled during hopping
    distance_driven: np.ndarray  # Distance traveled during driven motion
    velocity_trap: np.ndarray  # Velocity during hopping
    velocity_driven: np.ndarray  # Velocity during driven motion


def analyze(config: sim.SimulationConfig):
    '''tTot: maximum total amount of "cell time" this simulation is run
    n: number of particles shown
    p_driv: probability of driven motion as opposed to trap
    trap_size: size of trap (m)
    avg: average time between states
    dt: time step (s)
    '''
    exit_times = np.array([])
    max_n_steps = 0
    i = 0
    nout = 0
    flux_trap = np.array([], float)
    flux_driven = np.array([], float)
    distance_trap = np.array([], float)
    distance_driven = np.array([], float)
    velocity_trap = np.array([], float)
    velocity_driven = np.array([], float)
    all_x_trajectories = []
    all_y_trajectories = []

    for i in tqdm.tqdm(range(config.n_particles), desc="Running simulations to analyze flux, distance, velocities, MSD"):
        sim_output = sim.move(config, 2*np.pi*i/config.n_particles)
        max_n_steps = max(len(sim_output.x), max_n_steps)

        # Store flux
        fh = sum(sim_output.distance_trap) * len(sim_output.distance_trap)/max(1, len(sim_output.distance_trap)+len(sim_output.distance_driven)) # flux
        fd = sum(sim_output.distance_driven) * len(sim_output.distance_driven)/max(1, len(sim_output.distance_trap)+len(sim_output.distance_driven))
        flux_trap = np.append(flux_trap, fh/max(1, fd+fh))
        flux_driven = np.append(flux_driven, fd/max(1, fd+fh))

        # Store distances, velocities
        distance_trap = np.append(distance_trap, sum(sim_output.velocity_trap))
        distance_driven = np.append(distance_driven, sum(sim_output.velocity_driven))
        velocity_trap = np.concatenate((velocity_trap, sim_output.velocity_trap))
        velocity_driven = np.concatenate((velocity_driven, sim_output.velocity_driven))

        # Collect trajectories for MSD calculation
        all_x_trajectories.append(np.array(sim_output.x))
        all_y_trajectories.append(np.array(sim_output.y))

        if sim_output.exit_time != -1:
            nout += 1
            exit_times = np.append(exit_times, sim_output.exit_time)

        # Collect trajectories for MSD calculation
        all_x_trajectories.append(np.array(sim_output.x))
        all_y_trajectories.append(np.array(sim_output.y))


    msd = np.zeros(max_n_steps)
    for x_traj, y_traj in zip(all_x_trajectories, all_y_trajectories):
        # Pad trajectories if needed to match max_n_steps
        if len(x_traj) < max_n_steps:
            x_traj = np.pad(x_traj, (0, max_n_steps - len(x_traj)), 'edge')
            y_traj = np.pad(y_traj, (0, max_n_steps - len(y_traj)), 'edge')
        dx = x_traj - x_traj[0]
        dy = y_traj - y_traj[0]
        msd += dx**2 + dy**2

    msd /= max_n_steps
    print(f"{nout} out of {config.n_particles} exit the cell".format(nout, config.n_particles))
    print(f"Mean exit time: {np.mean(exit_times)}")
    print(f"Mean distance (hop): {np.mean(distance_trap)}, (driv): {np.mean(distance_driven)}")
    print(f"Mean flux (hop): {np.mean(flux_trap)}, (driv): {np.mean(flux_driven)}")
    print(f"Mean velocity (hop): {np.mean(velocity_trap)}, (driv): {np.mean(velocity_driven)}")

    return AnalysisOutput(
        msd=msd,
        exit_times=exit_times,
        max_n_steps=max_n_steps,
        flux_trap=flux_trap,
        flux_driven=flux_driven,
        distance_trap=distance_trap,
        distance_driven=distance_driven,
        velocity_trap=velocity_trap,
        velocity_driven=velocity_driven,
    )


def displacement_vs_time(
    total_time,
    n_particles,
    p_driv,
):
    x_all = []
    y_all = []
    config = sim.SimulationConfig(
        n_particles=n_particles,
        total_time=total_time,
        p_driv=p_driv,
        end_early=False,
    )

    for _ in tqdm.tqdm(range(n_particles), desc="Running simulations for displacement vs time"):
        sim_output = sim.move(config)
        x = np.concatenate([np.array([7.5]), sim_output.x * 1e6])  # Convert to micrometers
        y = np.concatenate([np.array([0]), sim_output.y * 1e6])
        x_all.append(x)
        y_all.append(y)

    # Find the maximum trajectory length
    max_len = max(len(x) for x in x_all)

    # Pad each trajectory with the last value to make them all the same length
    x_all_padded = [np.pad(x, (0, max_len - len(x)), mode='edge') for x in x_all]
    y_all_padded = [np.pad(y, (0, max_len - len(y)), mode='edge') for y in y_all]

    x_all = np.array(x_all_padded)
    y_all = np.array(y_all_padded)
    displacements = np.sqrt(x_all**2 + y_all**2)
    displacements -= 7.5 # Center the displacement around 0
    mean_dist = np.mean(displacements, axis=0)

    displacements = np.abs(displacements)
    mean_disp = np.mean(displacements, axis=0)
    mean_squared_disp = np.mean(displacements**2, axis=0)

    return mean_dist, mean_disp, mean_squared_disp


def plot_displacement_vs_time_varying_driven(
        p_driv_vals, # list of percentages of driven motion which should be tested
        n_particles,
        total_time = 600,
        dt = 0.001
):
    for p_driv in tqdm.tqdm(p_driv_vals, desc="Running simulations for each driven probability", unit="driven probabilities"):
        config = sim.SimulationConfig(
            n_particles=n_particles,
            total_time=total_time,
            p_driv=p_driv,
            end_early=False,
        )

        x_all = []
        y_all = []
        for _ in range(n_particles):
            sim_output = sim.move(config)
            x = np.concatenate([np.array([7.5]), sim_output.x * 1e6])  # Convert to micrometers
            y = np.concatenate([np.array([0]), sim_output.y * 1e6])
            x_all.append(x)
            y_all.append(y)

        # Find the maximum trajectory length
        max_len = max(len(x) for x in x_all)

        # Pad each trajectory with the last value to make them all the same length
        x_all_padded = np.array([np.pad(x, (0, max_len - len(x)), mode='edge') for x in x_all])
        y_all_padded = np.array([np.pad(y, (0, max_len - len(y)), mode='edge') for y in y_all])


        # Calculate mean signed displacement around 0
        displacements = np.sqrt(x_all_padded**2 + y_all_padded**2)
        displacements -= 7.5 # Center the displacement around 0
        mean_signed = np.mean(displacements, axis=0)

        # Set up time axis
        t = np.arange(len(mean_signed)) * dt    # time axis (s)

        # Plot
        plt.plot(t, mean_signed, label="⟨r − r₀⟩ (signed)")

