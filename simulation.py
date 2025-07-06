import argparse
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import random
from matplotlib import pyplot as plt
import statistics as st
import tqdm

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
    total_time: int = 2000
    n_particles: int = 1
    p_driv: float = 0.03
    trap_dist: float = TRAP_DIST
    trap_std: float = TRAP_STD
    time_between: float = TIME_BETWEEN_STATES
    dt: float = 0.001
    dirname: str = "sim"
    width: int = 600
    height: int = 600
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
    '''total_time: maximum total amount of "cell time" this simulation is run
    p_driv: probability of driven motion as opposed to trap
    trap_dist: distance between traps (m)
    time_between: average time between states'''
    k = random.uniform(1.5e-6, 2.6e-6) # spring constant; N/m

    x = [CELL_RADIUS/2 * np.cos(theta)]
    y = [CELL_RADIUS/2 * np.sin(theta)]
    x_center = x[0]
    y_center = y[0]

    # effective probability, b/c p_driv should be time-based and this program doesn't treat it as such
    # if p_driv != 0 and p_driv != 1:
    #     p_driv *= (time_between / (1 - p_driv + time_between*p_driv))

    def set_state_duration(d):
        inc = -1
        while inc < 0:
            inc = random.gauss(config.time_between, config.time_between)
        return inc

    # determine when particle changes states
    state_switch_times = []
    curr = random.random() < config.p_driv
    inc = set_state_duration(curr)
    driven = [curr]

    tempT = inc
    for i, t in enumerate(np.arange(0, config.total_time+2*config.dt, config.dt)):
        if t >= tempT:
            state_switch_times.append(i)  # TODO change to (i, is_driven) and get rid of `driven`
            curr = random.random() < config.p_driv
            inc = set_state_duration(curr)
            tempT = tempT + inc
        driven.append(curr)

    distance_trap = []
    distance_driven = []
    velocity_driven = []
    velocity_trap = []

    exit_time = -1
    start_of_state = 0
    reverse_direction = False

    frame_iterable = enumerate(np.arange(config.dt, config.total_time+config.dt, config.dt))
    if show_progress:
        frame_iterable = tqdm.tqdm(
            frame_iterable,
            total=config.total_time,
            unit="s",
            unit_scale=config.dt,
            desc="Time elapsed in simulation"
        )

    for i, t in frame_iterable:
        if driven[i]: # driven
            # set tentative coordinates
            x_new = 0
            y_new = 0

            # need to make sure particle doesn't go into nucleus
            while np.sqrt(x_new**2 + y_new**2) < NUCLEUS_RADIUS:
                dr = random.gauss(MOTOR_PROTEIN_SPEED*config.dt, MOTOR_PROTEIN_SPEED*config.dt)
                theta = np.arctan(y[-1]/x[-1])
                if x[-1] < 0:
                    theta += np.pi
                if reverse_direction:
                    theta += np.pi
                x_new = x[-1] - dr * np.cos(theta)
                y_new = y[-1] - dr * np.sin(theta)
            x.append(x_new + random.gauss(0, 2e-9))
            y.append(y_new + random.gauss(0, 2e-9))
            x_center = x[-1]
            y_center = y[-1]

        else: # trap
            # tentative coordinates
            x_new = 0
            y_new = 0
            
            # particle must not enter the nucleus
            while np.sqrt(x_new**2 + y_new**2) < NUCLEUS_RADIUS:
                dx = -k*(x[-1]-x_center)*config.dt/g + np.sqrt(2*D*config.dt) * random.gauss(0, 1)
                dy = -k*(y[-1]-y_center)*config.dt/g + np.sqrt(2*D*config.dt) * random.gauss(0, 1)
                x_new = x[-1] + dx
                y_new = y[-1] + dy

            x.append(x_new + random.gauss(0, 2e-9))
            y.append(y_new + random.gauss(0, 2e-9))

            # change states from trap to something else
            if i in state_switch_times:
                if not driven[i+1]:
                    dx = -x_center
                    dy = -y_center
                    while np.sqrt((x_center + dx)**2 + (y_center + dy)**2) < NUCLEUS_RADIUS:
                        new_center_dist = random.gauss(config.trap_dist, config.trap_std)
                        while new_center_dist < 0:
                            new_center_dist = random.gauss(config.trap_dist, config.trap_std)
                        rand = random.gauss(0, 0.4)
                        while np.abs(rand) > 1:
                            rand = random.gauss(0, 0.4)
                        theta = rand*2*np.pi
                        dx = new_center_dist * np.cos(theta)
                        dy = new_center_dist * np.sin(theta)
                    x_center += dx
                    y_center += dy

        # determine the direction of the next bout of driven motion
        if i in state_switch_times:
            reverse_direction = random.random()<0.5

        v_current = np.sqrt( (x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 ) / config.dt

        # store velocity information
        if driven[i]:
            velocity_driven.append(v_current)
        else:
            velocity_trap.append(v_current)

        # distance traveled over the course of a particular state
        if i in state_switch_times:
            tmp = np.sqrt((x[-1] - x[start_of_state])**2 + (y[-1] - y[start_of_state])**2)
            if driven[i]:
                distance_driven.append(tmp)
            else:
                distance_trap.append(tmp)
            start_of_state = i+1
        
        if np.sqrt(x[-1]**2+y[-1]**2) > CELL_RADIUS and exit_time == -1:
            exit_time = t
            if config.end_early: break

    return SimulationOutput(
        x=np.asarray(x),
        y=np.asarray(y),
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
