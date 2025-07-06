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


class SimInfo(NamedTuple):
    x: np.ndarray  # x-coordinates of the particle
    y: np.ndarray  # y-coordinates of the particle
    exit_time: float  # time at which the particle exits the cell
    distance_trap: np.ndarray  # distances traveled during hopping states
    distance_driven: np.ndarray  # distances traveled during driven states
    v_driven: np.ndarray  # velocities during driven states
    v_trap: np.ndarray  # velocities during hopping states

def move(
    total_time,
    p_driv,
    trap_dist,
    trap_std,
    time_between,
    theta,
    dt,
    eps=1e-9,
) -> SimInfo:
    '''total_time: maximum total amount of "cell time" this simulation is run
    p_driv: probability of driven motion as opposed to trap
    trap_dist: distance between traps (m)
    time_between: average time between states'''
    k = random.uniform(1.5e-6, 2.6e-6) # spring constant; N/m

    x = np.array([CELL_RADIUS/2 * np.cos(theta)])
    y = np.array([CELL_RADIUS/2 * np.sin(theta)])
    x_center = x[0]
    y_center = y[0]

    # effective probability, b/c p_driv should be time-based and this program doesn't treat it as such
    # if p_driv != 0 and p_driv != 1:
    #     p_driv *= (time_between / (1 - p_driv + time_between*p_driv))

    def set_state_duration(d):
        inc = -1
        while inc < 0:
            inc = random.gauss(time_between, time_between)
        return inc

    # determine when particle changes states
    changeTime = []
    curr = random.random() < p_driv
    inc = set_state_duration(curr)
    driven = [curr]

    tempT = inc
    for i, t in enumerate(np.arange(0, total_time+2*dt, dt)):
        if t >= tempT:
            changeTime.append(i)  # TODO change to (i, is_driven) and get rid of `driven`
            curr = random.random() < p_driv
            inc = set_state_duration(curr)
            tempT = tempT + inc
        driven.append(curr)

    exitTime = -1

    distance_trap = np.array([], float)
    distance_driven = np.array([], float)
    v_driven = np.array([], float)
    v_trap = np.array([], float)
    start_of_state = 0

    rev = False
    for i, t in (tqdm.tqdm(
        enumerate(np.arange(dt, total_time+dt, dt)),
        total=total_time,
        unit="s",
        unit_scale=dt,
        desc="Time elapsed in simulation"
    )):
        if driven[i]: # driven
            # set tentative coordinates
            x_new = 0
            y_new = 0

            # need to make sure particle doesn't go into nucleus
            while np.sqrt(x_new**2 + y_new**2) < NUCLEUS_RADIUS:
                dr = random.gauss(MOTOR_PROTEIN_SPEED*dt, MOTOR_PROTEIN_SPEED*dt)
                theta = np.arctan(y[-1]/x[-1])
                if x[-1] < 0:
                    theta += np.pi
                if rev:
                    theta += np.pi
                x_new = x[-1] - dr * np.cos(theta)
                y_new = y[-1] - dr * np.sin(theta)
            x = np.append(x, x_new + random.gauss(0, 2e-9))
            y = np.append(y, y_new + random.gauss(0, 2e-9))
            x_center = x[-1]
            y_center = y[-1]

        else: # trap
            # tentative coordinates
            x_new = 0
            y_new = 0
            
            # particle must not enter the nucleus
            while np.sqrt(x_new**2 + y_new**2) < NUCLEUS_RADIUS:
                dx = -k*(x[-1]-x_center)*dt/g + np.sqrt(2*D*dt) * random.gauss(0, 1)
                dy = -k*(y[-1]-y_center)*dt/g + np.sqrt(2*D*dt) * random.gauss(0, 1)
                x_new = x[-1] + dx
                y_new = y[-1] + dy

            x = np.append(x, x_new + random.gauss(0, 2e-9))
            y = np.append(y, y_new + random.gauss(0, 2e-9))

            # change states from trap to something else
            if i in changeTime:
                if not driven[i+1]:
                    dx = -x_center
                    dy = -y_center
                    while np.sqrt((x_center + dx)**2 + (y_center + dy)**2) < NUCLEUS_RADIUS:
                        new_center_dist = random.gauss(trap_dist, trap_std)
                        while new_center_dist < 0:
                            new_center_dist = random.gauss(trap_dist, trap_std)
                        rand = random.gauss(0, 0.4)
                        while np.abs(rand) > 1:
                            rand = random.gauss(0, 0.4)
                        theta = rand*2*np.pi
                        dx = new_center_dist * np.cos(theta)
                        dy = new_center_dist * np.sin(theta)
                    x_center += dx
                    y_center += dy

        # determine the direction of the next bout of driven motion
        if i in changeTime:
            rev = random.random()<0.5

        v_current = np.sqrt( (x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 ) / dt

        # store velocity information
        if driven[i]:
            v_driven = np.append(v_driven, v_current)
        else:
            v_trap = np.append(v_trap, v_current)

        # distance traveled over the course of a particular state
        if i in changeTime:
            tmp = np.sqrt((x[-1] - x[start_of_state])**2 + (y[-1] - y[start_of_state])**2)
            if driven[i]:
                distance_driven = np.append(distance_driven, tmp)
            else:
                distance_trap = np.append(distance_trap, tmp)
            start_of_state = i+1
        
        if np.sqrt(x[-1]**2+y[-1]**2) > CELL_RADIUS and exitTime == -1:
            exitTime = t
            break

    return x, y, exitTime, distance_trap, distance_driven, v_driven, v_trap


def graph(
    total_time,
    p_driv=0.03,
    trap_dist=TRAP_DIST,
    time_between=TIME_BETWEEN_STATES,
    theta=0,
    dt=1e-2,
):
    """Graph the movement of a particle in a cell."""
    x, y = move(
        total_time,
        p_driv=p_driv,
        trap_dist=trap_dist,
        time_between=time_between,
        theta=theta,
        dt=dt,
)
    gx = [i * 1e6 for i in x]
    gy = [i * 1e6 for i in y]
    for i in range(total_time-1):
        plt.scatter(gx[int(i/dt):int((i+1)/dt)], gy[int(i/dt):int((i+1)/dt)])
    centerx = []
    centery = []
    for i in range(total_time-1):
        centerx.append(st.mean(gx[int(i/dt):int((i+1)/dt)]))
        centery.append(st.mean(gy[int(i/dt):int((i+1)/dt)]))
    plt.plot(centerx, centery)
    plt.show()
