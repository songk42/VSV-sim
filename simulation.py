from typing import NamedTuple

import numpy as np
import random
from matplotlib import pyplot as plt
import statistics as st
import tqdm

eta = 0.006 # Pa-s
R = 8.6e-8 # m
Kb = 1.38064852e-23 # Boltzmann constant
T = 37 + 273.15 # K
g = 6 * np.pi * eta * R
D = Kb * T / g # diffusivity coefficient via Stokes-Einstein equation

CELL_RADIUS = 1.502e-5  # radius of cell (m)
NUCLEUS_RADIUS = 5e-6  # radius of cell nucleus (m)
TRAP_SIZE = 2.4e-7  # size of trap (m)
TRAP_DIST = 1.7e-7  # distance between traps (m)
TRAP_SD = 2.1e-7  # standard deviation of trap distance (m)
TIME_BETWEEN_STATES = 0.41  # average time between states (s)
MOTOR_PROTEIN_SPEED = 1e-6  # speed of motor proteins (m/s)


class SimInfo(NamedTuple):
    x: np.ndarray  # x-coordinates of the particle
    y: np.ndarray  # y-coordinates of the particle
    exit_time: float  # time at which the particle exits the cell
    hop: np.ndarray  # distances traveled during hopping states
    driv: np.ndarray  # distances traveled during driven states
    vd: np.ndarray  # velocities during driven states
    vh: np.ndarray  # velocities during hopping states

def move(
    total_time,
    pDriv=0.03,
    trap_dist=TRAP_DIST,
    time_between=TIME_BETWEEN_STATES,
    theta=0,
    dt=1e-2,
) -> SimInfo:
    '''total_time: maximum total amount of "cell time" this simulation is run
    pDriv: probability of driven motion as opposed to trap
    trap_dist: distance between traps (m)
    time_between: average time between states'''
    k = random.uniform(6e-7, 2.6e-6) # spring constant; N/m

    x = np.array([CELL_RADIUS/2 * np.cos(theta)])
    y = np.array([CELL_RADIUS/2 * np.sin(theta)])
    xC = x[0]
    yC = y[0]

    trap_radius = trap_dist / 2

    # effective probability, b/c pDriv should be time-based and this program doesn't treat it as such
    if pDriv != 0 and pDriv != 1:
        pDriv *= (time_between / (1 - pDriv + time_between*pDriv))

    def set_state_duration(d):
        inc = -1
        while inc < 0:
            if d:
                inc = random.gauss(1, 1)
            else:
                inc = random.gauss(time_between, time_between)
        return inc

    # determine when particle changes states
    changeTime = np.array([], float)
    driven = {}
    curr = random.random() < pDriv
    inc = set_state_duration(curr)
    driven[0] = curr

    tempT = np.round(inc, 2)
    for i, t in enumerate(np.arange(0, total_time+2*dt, dt)):
        if t >= tempT:
            changeTime = np.append(changeTime, tempT)
            curr = random.random() < pDriv
            inc = set_state_duration(curr)
            tempT = np.round(tempT + inc, 2)
        driven[np.round(t, 2)] = curr

    exitTime = -1

    hop = np.array([], float)
    driv = np.array([], float)
    vd = np.array([], float)
    vh = np.array([], float)
    prev = 0

    rev = False
    for t in tqdm.tqdm(np.arange(dt, total_time+dt, dt)):
        t = np.round(t, 2)
        if driven[t]: # driven
            dr = random.gauss(MOTOR_PROTEIN_SPEED*dt, MOTOR_PROTEIN_SPEED*dt)
            m = np.sqrt(x[-1]**2 + y[-1]**2)
            # set direction (honestly probably not necessary every single time)
            # I'm not convinced that m ever reaches 0
            if m < 1e-10:
                theta = 2 * np.pi * random.random()
            else:
                theta = np.arctan(y[-1]/x[-1])
                if x[-1] < 0:
                    theta += np.pi
                if rev:
                    theta += np.pi

            # set tentative coordinates
            xtemp = x[-1] + dr * np.cos(theta)
            ytemp = y[-1] + dr * np.sin(theta)

            # need to make sure particle doesn't go into nucleus
            if np.sqrt(xtemp**2 + ytemp**2) >= NUCLEUS_RADIUS:
                x = np.append(x, xtemp)
                y = np.append(y, ytemp)
                xC = x[-1]
                yC = y[-1]
            else:
                return x, y, True, exitTime, hop, driv, vd, vh

        else: # trap
            # tentative coordinates
            xtemp = x[-1] - k*(x[-1]-xC)*dt/g + np.sqrt(2*D*dt) * random.gauss(0, 1)
            ytemp = y[-1] - k*(y[-1]-yC)*dt/g + np.sqrt(2*D*dt) * random.gauss(0, 1)
            
            # particle must not enter the nucleus
            if np.sqrt(xtemp**2 + ytemp**2) < NUCLEUS_RADIUS:
                return x, y, True, exitTime, hop, driv, vd, vh
            
            x = np.append(x, xtemp)
            y = np.append(y, ytemp)

            # change states from trap to something else
            if np.round(t, 2) in changeTime:
                if not driven[np.round(t+dt, 2)]:
                    dr = -1
                    while dr < 0:
                        dr = random.gauss(TRAP_DIST, TRAP_SD)
                    rand = 2
                    while np.abs(rand) > 1:
                        rand = random.gauss(0, 0.4)
                    theta = rand*2*np.pi
                    xC += dr * np.cos(theta)
                    yC += dr * np.sin(theta)

        # # halve any differences (idk if this is a good idea, but it's a quick fix)
        # x[-1] = (x[-1] + x[-2])/2
        # y[-1] = (y[-1] + y[-2])/2

        # noise
        x[-1] += random.gauss(0, 2e-9)
        y[-1] += random.gauss(0, 2e-9)

        # determine the direction of the next bout of driven motion
        if np.round(t, 2) in changeTime:
            rev = random.random()<0.5

        veltmp = np.sqrt( (x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 ) / dt

        # store velocity information
        if driven[np.round(t, 2)]:
            vd = np.append(vd, veltmp)
        else:
            vh = np.append(vh, veltmp)

        # distance traveled over the course of a particular state
        if np.round(t, 2) in changeTime:
            tmp = np.sqrt((x[-1] - x[prev])**2 + (y[-1] - y[prev])**2)
            if driven[np.round(t, 2)]:
                driv = np.append(driv, tmp)
            else:
                hop = np.append(hop, tmp)
            prev = len(x)
        
        if np.sqrt(x[-1]**2+y[-1]**2) > CELL_RADIUS and exitTime == -1:
            exitTime = t
            break

    return x, y, False, exitTime, hop, driv, vd, vh


def graph(
    total_time,
    pDriv=0.03,
    trap_dist=TRAP_DIST,
    time_between=TIME_BETWEEN_STATES,
    theta=0,
    dt=1e-2,
):
    """Graph the movement of a particle in a cell."""
    x, y = move(
        total_time,
        pDriv=pDriv,
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
