import argparse
import time
from dataclasses import dataclass
from functools import wraps
from typing import NamedTuple

import numpy as np
import random
from matplotlib import pyplot as plt
import statistics as st
from tqdm import tqdm
from numba import njit
from numba.typed import List

eta = 0.006 # Pa-s
R = 8.6e-8 # radius of particle (m)
Kb = 1.38064852e-23 # Boltzmann constant
T = 37 + 273.15 # K
g = 6 * np.pi * eta * R  # viscous drag coefficient
D = Kb * T / g # diffusivity coefficient via Stokes-Einstein equation
k = random.uniform(1.5e-6, 2.6e-6) # spring constant; N/m

CELL_RADIUS = 1.502e-5  # radius of cell (m)
NUCLEUS_RADIUS = 5e-6  # radius of cell nucleus (m)
TRAP_SIZE = 2.4e-7  # size of trap (m)
TRAP_DIST = 1.7e-7  # distance between traps (m)
TRAP_STD = 2.1e-7  # standard deviation of trap distance (m)
TIME_BETWEEN_STATES = 0.41  # average time between states (s)
MOTOR_PROTEIN_SPEED = 1e-6  # speed of motor proteins (m/s)


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__!r} took {end - start:.4f} seconds")
        return result
    return wrapper


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


@njit(cache=True)
def truncated_gauss(mu, sigma, low=-np.inf, high=np.inf):
    """Gaussian sample truncated to [low, high] – numba compatible."""
    while True:
        x = np.random.normal(mu, sigma)
        if low <= x <= high:
            return x

@njit(cache=True)
def outside_nucleus(x, y):
    return x * x + y * y > NUCLEUS_RADIUS ** 2

@njit(cache=True)
def calc_directed_step(current_x, current_y, reverse_direction, dt):
    dr = np.random.normal(MOTOR_PROTEIN_SPEED * dt, MOTOR_PROTEIN_SPEED * dt)

    theta = np.arctan2(current_y, current_x)
    if reverse_direction:
        theta += np.pi

    jitter = 2e-9
    while True:
        nx = current_x + dr * np.cos(theta) + np.random.normal(0.0, jitter)
        ny = current_y + dr * np.sin(theta) + np.random.normal(0.0, jitter)
        if outside_nucleus(nx, ny):
            return nx, ny


@njit(cache=True)
def calc_diffusive_step(current_x, current_y, trap_x_center, trap_y_center, dt):
    # 1. Restoring‐force (drift) displacement
    displacement_from_center_x = current_x - trap_x_center
    displacement_from_center_y = current_y - trap_y_center

    trap_force_x = -k * displacement_from_center_x
    trap_force_y = -k * displacement_from_center_y

    drift_disp_x = (trap_force_x / g) * dt
    drift_disp_y = (trap_force_y / g) * dt

    # 2. Thermal‐diffusion displacement
    diffusion_std = np.sqrt(2 * D * dt)

    while True:
        diffusion_disp_x = diffusion_std * np.random.normal(0, 1)
        diffusion_disp_y = diffusion_std * np.random.normal(0, 1)

        # 3. Fine‐scale jitter
        jitter_std = 2e-9  # meters

        jitter_disp_x = np.random.normal(0, jitter_std)
        jitter_disp_y = np.random.normal(0, jitter_std)

        # 4. Sum all contributions into the total step
        dx = drift_disp_x + diffusion_disp_x + jitter_disp_x
        dy = drift_disp_y + diffusion_disp_y + jitter_disp_y

        # 5. Compute the candidate next position
        next_x = current_x + dx
        next_y = current_y + dy

        # 6. Only accept if it's outside the nucleus
        if outside_nucleus(next_x, next_y):
            return next_x, next_y

@njit(cache=True)
def calc_new_trap_position(trap_x, trap_y, trap_dist, trap_std):
    while True:
        # 1. calculate random direction and distance
        r = truncated_gauss(trap_dist, trap_std, 0.0, np.inf)
        theta = np.random.uniform(0.0, 2.0 * np.pi)

        # 2. propose new center coordinates
        new_trap_x = trap_x + r * np.cos(theta)
        new_trap_y = trap_y + r * np.sin(theta)

        # 3. only accept if it's outside the nucleus
        if outside_nucleus(new_trap_x, new_trap_y):
            return new_trap_x, new_trap_y

@njit(cache=True)
def generate_state_duration(time_between_state_changes):
    while True:
        dur = np.random.normal(time_between_state_changes, time_between_state_changes)

        if dur > 0.0:
            return dur

@njit(cache=True)
def _move(total_time, dt,
          p_driv, avg_state_duration,
          trap_dist, trap_std, theta, stop_on_cell_exit):

    # Pre-allocate trajectory arrays
    max_states = int(total_time / dt) + 1
    x_history  = np.empty(max_states, dtype=np.float64)
    y_history  = np.empty(max_states, dtype=np.float64)
    final_i = max_states - 1

    # Initialize fixed-size numpy buffers
    diffusive_distance = np.empty(max_states,  dtype=np.float64)
    driven_distance = np.empty_like(diffusive_distance)
    diffusive_velocity = np.empty_like(diffusive_distance)
    driven_velocity = np.empty_like(diffusive_distance)

    # Buffer indices we're writing to (replacing expensive .append calls)
    diffusive_distance_index = 0
    driven_distance_index  = 0
    diffusive_velocity_index  = 0
    driven_velocity_index   = 0

    # Initial positions (particles evenly spaced out halfway between nucleus and cell membrane)
    start_radius = CELL_RADIUS / 2
    x_history[0] = trap_x = start_radius * np.cos(theta)
    y_history[0] = trap_y = start_radius * np.sin(theta)

    # generate state schedule on the fly
    is_driven         = np.random.random() < p_driv # Current state
    next_switch_time  = generate_state_duration(avg_state_duration)
    state_start_index = 0 # Index current state began
    reverse_next      = False
    exit_time         = -1.0

    # MAIN SIMULATION LOOP
    for i in range(1, max_states):
        current_time = i * dt

        # Check for state change
        if current_time >= next_switch_time:

            # Record distance traveled in state
            dx = x_history[i-1] - x_history[state_start_index]
            dy = y_history[i-1] - y_history[state_start_index]
            dist = np.hypot(dx, dy)
            if is_driven:
                driven_distance[driven_distance_index] = dist
                driven_distance_index += 1
            else:
                diffusive_distance[diffusive_distance_index] = dist
                diffusive_distance_index += 1

            # Reset state tracking, decide next state
            state_start_index = i - 1
            is_driven = np.random.random() < p_driv

            # Schedule next state change
            next_switch_time += generate_state_duration(avg_state_duration)
            reverse_next = np.random.random() < 0.5 # 50% chance to reverse

            # If just switched to diffusive state, move trap
            if not is_driven:
                # Particle could have been in a diffusive state before as well,
                # so we'd consider this the particle escaping to a new trap
                trap_x, trap_y = calc_new_trap_position(
                    trap_x, trap_y, trap_dist, trap_std
                )

        # Calculate next position based on current state
        if is_driven:
            new_x, new_y = calc_directed_step(x_history[i-1], y_history[i-1], reverse_next, dt)
        else:
            new_x, new_y = calc_diffusive_step(x_history[i-1], y_history[i-1], trap_x, trap_y, dt)

        # During driven motion, trap follows particle
        if is_driven:
            trap_x, trap_y = new_x, new_y

        # Record position
        x_history[i] = new_x
        y_history[i] = new_y

        # Record velocity by state type
        v_now = np.hypot(new_x - x_history[i-1], new_y - y_history[i-1]) / dt
        if is_driven:
            driven_velocity[driven_velocity_index] = v_now
            driven_velocity_index += 1
        else:
            diffusive_velocity[diffusive_velocity_index] = v_now
            diffusive_velocity_index += 1

        # Check for cell exit
        if np.hypot(new_x, new_y) > CELL_RADIUS:
            exit_time = current_time
            final_i   = i

            if stop_on_cell_exit:
                break

    return (x_history[:final_i + 1],
                         y_history[:final_i + 1],
                         exit_time,
                         diffusive_distance[:diffusive_distance_index].copy(),
                         driven_distance[:driven_distance_index].copy(),
                         driven_velocity[:driven_velocity_index].copy(),
                         diffusive_velocity[:diffusive_velocity_index].copy())

def move(config, theta: float = 0.0, stop_on_cell_exit = True):
    """Acts as a wrapper for _move, passing in arguments
    so the njit doesn't need to handle SimulationConfig objects

    Returns:
        SimulationOutput: A named tuple containing:
            x (np.ndarray): x-coordinates of the particle
            y (np.ndarray): y-coordinates of the particle
            exit_time (float): Time at which particle exits cell (-1 if doesn't exit)
            distance_trap (list): Distances traveled during hopping states
            distance_driven (list): Distances traveled during driven states
            velocity_driven (list): Velocities during driven states
            velocity_trap (list): Velocities during hopping states
    """

    x, y, exit_time, dist_trap, distance_driven, vel_driven, vel_trap = _move(
        config.total_time, config.dt,
        config.p_driv, config.time_between,
        config.trap_dist, config.trap_std, theta, stop_on_cell_exit
    )

    return SimulationOutput(
        x=np.array(x),
        y=np.array(y),
        exit_time=exit_time,
        distance_trap=np.array(dist_trap),
        distance_driven=np.array(distance_driven),
        velocity_driven=np.array(vel_driven),
        velocity_trap=np.array(vel_trap)
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

