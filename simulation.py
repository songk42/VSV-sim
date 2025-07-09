import argparse
import time
from dataclasses import dataclass
from functools import wraps
from typing import NamedTuple

import numpy as np
import random
from matplotlib import pyplot as plt
import statistics as st
import tqdm
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
        if x >= low and x <= high:
            return x

@njit(cache=True)
def outside_nucleus(x, y):
    return x * x + y * y > NUCLEUS_RADIUS ** 2

@njit(cache=True)
def calc_directed_step(curr_x, curr_y, reverse_direction, dt):
    dr = np.random.normal(MOTOR_PROTEIN_SPEED * dt, MOTOR_PROTEIN_SPEED * dt)

    theta = np.arctan2(curr_y, curr_x)
    if reverse_direction:
        theta += np.pi

    jitter = 2e-9
    while True:
        nx = curr_x + dr * np.cos(theta) + np.random.normal(0.0, jitter)
        ny = curr_y + dr * np.sin(theta) + np.random.normal(0.0, jitter)
        if outside_nucleus(nx, ny):
            return nx, ny      # <-- tuple instead of Point


@njit(cache=True)
def calc_diffusive_step(curr_x, curr_y,
                        trap_x, trap_y,
                        dt):
    diff_std   = np.sqrt(2.0 * D * dt)
    jitter_std = 2e-9

    fx = -k * (curr_x - trap_x)
    fy = -k * (curr_y - trap_y)
    drift_x = (fx / g) * dt
    drift_y = (fy / g) * dt

    while True:
        nx = curr_x + drift_x + diff_std * np.random.normal() \
             + np.random.normal(0.0, jitter_std)
        ny = curr_y + drift_y + diff_std * np.random.normal() \
             + np.random.normal(0.0, jitter_std)
        if outside_nucleus(nx, ny):
            return nx, ny      # tuple instead of Point

@njit(cache=True)
def calc_new_trap_position(trap_x, trap_y, trap_dist, trap_std):
    while True:
        r     = truncated_gauss(trap_dist, trap_std, 0.0, np.inf)
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        nx    = trap_x + r * np.cos(theta)
        ny    = trap_y + r * np.sin(theta)
        if outside_nucleus(nx, ny):
            return nx, ny

@njit(cache=True)
def set_state_duration(mean_len):
    dur = -1.0
    while dur < 0.0:
        dur = np.random.normal(mean_len, mean_len)
    return dur


@njit(nopython=True, cache=True)
def _move(total_time, dt,
          p_driv, mean_len,
          trap_dist, trap_std, theta):

    n_steps = int(total_time / dt) + 1
    x_hist  = np.empty(n_steps, dtype=np.float64)
    y_hist  = np.empty(n_steps, dtype=np.float64)

    # variable-length typed lists
    dist_trap   = List.empty_list(np.float64)
    dist_driven = List.empty_list(np.float64)
    vel_trap    = List.empty_list(np.float64)
    vel_driven  = List.empty_list(np.float64)

    # Initial positions
    x_hist[0] = trap_x = 0.5 * CELL_RADIUS * np.cos(theta)
    y_hist[0] = trap_y = 0.5 * CELL_RADIUS * np.sin(theta)

    # generate state schedule on the fly
    driven_prev   = np.random.random() < p_driv
    switch_time   = set_state_duration(mean_len)
    start_idx     = 0
    reverse_next  = False
    exit_time     = -1.0
    final_i       = 0

    for i in range(1, n_steps):
        t = i * dt

        # change state?
        if t >= switch_time:
            # finish previous run
            dist = np.hypot(x_hist[i-1] - x_hist[start_idx],
                            y_hist[i-1] - y_hist[start_idx])
            if driven_prev:
                dist_driven.append(dist)
            else:
                dist_trap.append(dist)

            start_idx = i - 1
            driven_prev = np.random.random() < p_driv
            switch_time += set_state_duration(mean_len)
            reverse_next = np.random.random() < 0.5

            # trap-to-trap hop
            if not driven_prev:
                trap_x, trap_y = calc_new_trap_position(
                    trap_x, trap_y, trap_dist, trap_std)

        # integrate one step
        if driven_prev:
            nx, ny = calc_directed_step(
                x_hist[i-1], y_hist[i-1], reverse_next, dt)
        else:
            nx, ny = calc_diffusive_step(
                x_hist[i-1], y_hist[i-1], trap_x, trap_y, dt)

        # trap follows during driven periods
        if driven_prev:
            trap_x, trap_y = nx, ny

        x_hist[i] = nx
        y_hist[i] = ny

        v_now = np.hypot(nx - x_hist[i-1], ny - y_hist[i-1]) / dt
        if driven_prev:
            vel_driven.append(v_now)
        else:
            vel_trap.append(v_now)

        # left the cell?
        if np.hypot(nx, ny) > CELL_RADIUS:
            exit_time = t
            final_i   = i
            break
    else:
        final_i = n_steps - 1

    return (x_hist[:final_i + 1],
            y_hist[:final_i + 1],
            exit_time,
            dist_trap,
            dist_driven,
            vel_driven,
            vel_trap)


def move(config, theta: float = 0.0):
    """Acts as a wrapper for _move, passing in arguments
    so the njit doesn't need to handle SimulationConfig objects"""
    return _move(config.total_time, config.dt,
                 config.p_driv, config.time_between,
                 config.trap_dist, config.trap_std, theta)


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

