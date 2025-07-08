import argparse
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import random
from matplotlib import pyplot as plt
import statistics as st
import tqdm
from numba import njit

from point import Point

eta = 0.006 # Pa-s
R = 8.6e-8 # radius of particle (m)
Kb = 1.38064852e-23 # Boltzmann constant
T = 37 + 273.15 # K
g = 6 * np.pi * eta * R  # viscous drag coefficient
D = Kb * T / g # diffusivity coefficient via Stokes-Einstein equation

CELL_RADIUS = 1.502e-5  # radius of cell (m)
NUCLEUS_RADIUS = 5e-6  # radius of cell nucleus (m)
TRAP_SIZE = 2.4e-7  # size of trap (m)
TRAP_DIST = 1.7e-7  # distance between traps (m)
TRAP_STD = 2.1e-7  # standard deviation of trap distance (m)
TIME_BETWEEN_STATES = 0.41  # average time between states (s)
MOTOR_PROTEIN_SPEED = 1e-6  # speed of motor proteins (m/s)

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    total_time: int = 2000  # maximum simulation time (s)
    n_particles: int = 1
    p_driv: float = 1  # probability of driven motion (0.0-1.0) should be 0.03
    trap_dist: float = TRAP_DIST  # distance between traps (m)
    trap_std: float = TRAP_STD  # standard deviation of trap distance (m)
    time_between: float = TIME_BETWEEN_STATES
    dt: float = 0.001  # time step (s)
    dirname: str = "sim"
    width: int = 600  # canvas width in pixels
    height: int = 600  # canvas height in pixels
    record_frames: bool = False
    end_early: bool = True  # whether to end the simulation early if the particle exits the cell

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SimulationConfig':
        """Create configuration from command line arguments"""
        return cls(
            total_time=args.total_time,
            n_particles=args.n_particles,
            p_driv=args.p_driv,
            trap_dist=args.trap_dist,
            time_between=args.time_between,
            dt=args.dt,
            dirname=args.dirname,
            width=args.width,
            height=args.height,
            record_frames=args.record_frames,
        )


class SimulationOutput(NamedTuple):
    x: np.ndarray  # x-coordinates of the particle
    y: np.ndarray  # y-coordinates of the particle
    exit_time: float  # time at which the particle exits the cell
    distance_trap: np.ndarray  # distances traveled during hopping states
    distance_driven: np.ndarray  # distances traveled during driven states
    velocity_driven: np.ndarray  # velocities during driven states
    velocity_trap: np.ndarray  # velocities during hopping states


def move(
        config: SimulationConfig,
        theta: float = 0,
        show_progress: bool = True,
) -> SimulationOutput:
    '''
    Simulate the movement of a particle in a cell.
    Parameters:
        config: SimulationConfig object containing simulation parameters.
        theta: Initial angle of the particle in radians (default is 0).
        show_progress: Whether to show a progress bar (default is True).
    Returns:
        SimulationOutput: Contains the x and y coordinates of the particle,
        the time at which the particle exits the cell, distances traveled during
        hopping and driven states, and velocities during those states.
    '''

    def calc_directed_step(current_x, current_y, reverse_direction):
        dr = random.gauss(MOTOR_PROTEIN_SPEED*config.dt, MOTOR_PROTEIN_SPEED*config.dt)
        theta = np.arctan(current_y/current_x)
        if current_x < 0:
            theta += np.pi
        if reverse_direction:
            theta += np.pi
        next_x = current_x + dr * np.cos(theta) + random.gauss(0, 2e-9)
        next_y = current_y + dr * np.sin(theta) + random.gauss(0, 2e-9)

        if outside_nucleus(next_x, next_y):
            return Point(next_x, next_y)
        else: # Recursively calculate directed steps until one is outside nucleus
            return calc_directed_step(current_x, current_y, reverse_direction)


    def calc_diffusive_step(current_x, current_y, trap_x_center, trap_y_center):
        # 1. Restoring‐force (drift) displacement
        displacement_from_center_x = current_x - trap_x_center
        displacement_from_center_y = current_y - trap_y_center

        trap_force_x = -k * displacement_from_center_x
        trap_force_y = -k * displacement_from_center_y

        drift_disp_x = (trap_force_x / g) * config.dt
        drift_disp_y = (trap_force_y / g) * config.dt

        # 2. Thermal‐diffusion displacement
        diffusion_std = np.sqrt(2 * D * config.dt)

        while True:
            diffusion_disp_x = diffusion_std * random.gauss(0, 1)
            diffusion_disp_y = diffusion_std * random.gauss(0, 1)

            # 3. Fine‐scale jitter
            jitter_std = 2e-9  # meters

            jitter_disp_x = random.gauss(0, jitter_std)
            jitter_disp_y = random.gauss(0, jitter_std)

            # 4. Sum all contributions into the total step
            dx = drift_disp_x + diffusion_disp_x + jitter_disp_x
            dy = drift_disp_y + diffusion_disp_y + jitter_disp_y

            # 5. Compute the candidate next position
            next_x = current_x + dx
            next_y = current_y + dy

            if outside_nucleus(next_x, next_y):
                return Point(next_x, next_y)


    def calc_new_trap_position(trap_x, trap_y):
        while True:
            # 1. calculate random direction and distance
            r = truncated_gauss(config.trap_dist, config.trap_std, low=0)
            theta = truncated_gauss(0,0.8 * np.pi,low=-2 * np.pi,high=2 * np.pi)

            # 2. propose new center coordinates
            new_trap_x = trap_x + r * np.cos(theta)
            new_trap_y = trap_y + r * np.sin(theta)

            # 3. only accept if it's outside the nucleus
            if outside_nucleus(new_trap_x, new_trap_y):
                return Point(new_trap_x, new_trap_y)


    k = random.uniform(1.5e-6, 2.6e-6) # spring constant; N/m

    n_steps = int(config.total_time/config.dt) + 1

    x_history = np.empty(n_steps, dtype=np.float64)
    y_history = np.empty(n_steps, dtype=np.float64)

    next_x = trap_x_center = x_history[0] = CELL_RADIUS/2 * np.cos(theta)
    next_y = trap_y_center = y_history[0] = CELL_RADIUS/2 * np.sin(theta)

    # effective probability, b/c p_driv should be time-based and this program doesn't treat it as such
    # if p_driv != 0 and p_driv != 1:
    #     p_driv *= (time_between / (1 - p_driv + time_between*p_driv))

    def set_state_duration(d):
        state_length = -1
        while state_length < 0:
            state_length = random.gauss(config.time_between, config.time_between)
        return state_length

    # determine when particle changes states
    state_switch_times = []
    curr = random.random() < config.p_driv
    state_length = set_state_duration(curr)
    driven = [curr]

    state_end_time = state_length
    for i, t in enumerate(np.arange(0, config.total_time+2*config.dt, config.dt)):
        if t >= state_end_time:
            state_switch_times.append(i)
            curr = random.random() < config.p_driv
            state_length = set_state_duration(curr)
            state_end_time = state_end_time + state_length
        driven.append(curr)

    distance_trap = []
    distance_driven = []
    velocity_driven = []
    velocity_trap = []

    exit_time = -1
    start_of_state = 0
    reverse_direction = False

    def outside_nucleus(x, y):
        return x**2 + y**2 > NUCLEUS_RADIUS**2


    frame_iterable = enumerate(np.arange(config.dt, config.total_time+config.dt, config.dt))
    if show_progress:
        frame_iterable = tqdm.tqdm(
            frame_iterable,
            total=config.total_time,
            unit=" virtual s",
            unit_scale=config.dt,
            desc="Time elapsed in simulation"
        )

    for i, t in frame_iterable:
        # Update current_x, current_y
        current_x = next_x
        current_y = next_y

        # Calc next position
        if driven[i]:
            next_pos = calc_directed_step(current_x,current_y,reverse_direction=reverse_direction)
        else:
            next_pos = calc_diffusive_step(current_x,current_y,trap_x_center=trap_x_center, trap_y_center=trap_y_center)
        next_x = next_pos.x
        next_y = next_pos.y

        # Record particle position
        x_history[i+1] = next_x
        y_history[i+1] = next_y

        # Update trap center
        # If driven, have trap follow particle
        if driven[i]:
            trap_x_center = next_x
            trap_y_center = next_y

        # If in active diffusion, only move trap if particle just escaped its trap
        # We say particles escape their trap when a state_switch_time occurs and both the original and new state are active diffusion
        state_just_switched = i in state_switch_times
        next_state_will_be_trap_state = not driven[i+1]
        particle_just_escaped_trap = not driven[i] and state_just_switched and next_state_will_be_trap_state
        if particle_just_escaped_trap:
            new_trap_center = calc_new_trap_position(trap_x_center, trap_y_center)
            trap_x_center = new_trap_center.x
            trap_y_center = new_trap_center.y

        # determine the direction of the next bout of driven motion
        if state_just_switched:
            reverse_direction = random.random()<0.5

        v_current = np.sqrt( (next_x - current_x)**2 + (next_y - current_y)**2 ) / config.dt

        # store velocity information
        if driven[i]:
            velocity_driven.append(v_current)
        else:
            velocity_trap.append(v_current)

        # distance traveled over the course of a particular state
        if i in state_switch_times:
            dist = np.sqrt((next_x - x_history[start_of_state])**2 + (next_y - y_history[start_of_state])**2)
            if driven[i]:
                distance_driven.append(dist)
            else:
                distance_trap.append(dist)
            start_of_state = i+1

        # Exit if particle escape cell
        if np.sqrt(next_x**2+next_y**2) > CELL_RADIUS:
            exit_time = t
            if config.end_early: break



    return SimulationOutput(
        x=np.asarray(x_history),
        y=np.asarray(y_history),
        exit_time=exit_time,
        distance_trap=np.asarray(distance_trap),
        distance_driven=np.asarray(distance_driven),
        velocity_driven=np.asarray(velocity_driven),
        velocity_trap=np.asarray(velocity_trap),
    )


def graph(
        config: SimulationConfig,
        theta: float = 0,
):
    """Graph the movement of a particle in a cell."""
    sim_output = move(config, theta)
    x = sim_output.x
    y = sim_output.y

    gx = [i * 1e6 for i in x]
    gy = [i * 1e6 for i in y]
    for i in range(config.total_time-1):
        plt.scatter(gx[int(i/config.dt):int((i+1)/config.dt)], gy[int(i/config.dt):int((i+1)/config.dt)])
    centerx = []
    centery = []
    for i in range(config.total_time-1):
        centerx.append(st.mean(gx[int(i/config.dt):int((i+1)/config.dt)]))
        centery.append(st.mean(gy[int(i/config.dt):int((i+1)/config.dt)]))
    plt.plot(centerx, centery)
    plt.show()

@njit
def truncated_gauss(mu, sigma, low=None, high=None):
    """Return a Gaussian(μ,σ) sample allowing truncating >= low and/or <= high."""
    while True:
        x = random.gauss(mu, sigma)
        if (low is None or x >= low) and (high is None or x <= high):
            return x