from typing import NamedTuple
import numpy as np
import random
from matplotlib import pyplot as plt
import statistics as st
import tkinter as tk
import PIL.ImageGrab as ImageGrab
import time
import os
import csv
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


class Simulation:
    def __init__(
        self,
        total_time,
        n_particles=1,
        pDriv=0.03,
        trap_dist=TRAP_DIST,
        time_between=TIME_BETWEEN_STATES,
        dt=0.01,
        dirname="sim",
        width=600,
        height=600,
        write_to_ps=False,
    ):
        '''
        total_time: maximum total amount of "cell time" this simulation is run
        n_particles: number of particles shown
        pDriv: probability of driven motion as opposed to trap
        trap_dist: size of trap (m)
        time_between: average time between states
        dt: time step for simulation
        dirname: directory to save simulation files
        width: width of canvas
        height: height of canvas
        write_to_ps: if true, write canvas as Postscript files
        '''
        self.root = tk.Tk()

        self.width = width
        self.height = height
    
        self.radius_x = self.width / 2
        self.radius_y = self.height / 2

        ### simulation settings ###
        self.n_particles = n_particles
        self.pDriv = pDriv
        self.trap_dist = trap_dist
        self.time_between = time_between
        self.total_time = total_time
        self.dt = dt
        self.coords = []
        self.vsv = []  # TODO difference between coords and vsv?

        self.frame_number = 1  # current frame number for simulation

        ### canvas settings ###
        self.dirname = dirname
        self.write_to_ps = write_to_ps
        try:
            os.mkdir(self.dirname)
        except:
            pass

        self.scale_factor = 1e7  # scale factor for the canvas

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='white')
        self.canvas.grid(row=0, columnspan=3)
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill='black')
        # cell, nucleus boundaries
        self.canvas.create_oval(
            self.radius_x-CELL_RADIUS*self.scale_factor,
            self.radius_y-CELL_RADIUS*self.scale_factor,
            self.radius_x+CELL_RADIUS*self.scale_factor,
            self.radius_y+CELL_RADIUS*self.scale_factor,
            outline='yellow'
        )
        self.canvas.create_oval(
            self.radius_x-NUCLEUS_RADIUS*self.scale_factor,
            self.radius_y-NUCLEUS_RADIUS*self.scale_factor,
            self.radius_x+NUCLEUS_RADIUS*self.scale_factor,
            self.radius_y+NUCLEUS_RADIUS*self.scale_factor,
            outline='white'
        )

        self.time_label = tk.Label(self.root, font=('arial', '30'))
        self.time_label.grid(row=1, column=1)

        self.bot = tk.Frame()
        self.bot.grid(row=2, columnspan=3)

        self.pausetxt = tk.StringVar()
        self.pausetxt.set("Play")
        self.playing = tk.BooleanVar()
        self.playing.set(False)

        self.pauseB = tk.Button(self.bot, text='Play', command=self.toggle, font=('arial', '30'))
        self.pauseB.grid(row=0, column=0)

        self.resetB = tk.Button(self.bot, text='Reset', command=self.reset_canvas, font=('arial', '30'))
        self.resetB.grid(row=0, column=1)

        self.csvB = tk.Button(self.bot, text='Export to CSV', command=self.writeToFile, font=('arial', '30'))
        self.csvB.grid(row=0, column=2)

    # Trails: tag all lines so we can clear on reset
    # code from @BenHoffman06
    def draw_trail(self, prev_x, prev_y, new_x, new_y, color='white'):
        self.canvas.create_line(prev_x, prev_y, new_x, new_y,
                           fill=color, width=1, tags='trail')

    # Reset trails and animation
    def reset_canvas(self):
        self.canvas.delete('trail')
        for idx in range(self.n_particles):
            curr = self.canvas.coords(self.vsv[idx])
            target_x = self.radius_x + self.coords[idx][0][0]
            target_y = self.radius_y + self.coords[idx][1][0]
            self.canvas.move(self.vsv[idx], target_x - curr[0], target_y - curr[1])
        self.frame_number = 1
        self.time_label.config(text='Time:  0.00 s')
        self.canvas.update()

    def save_canvas(self, fname):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        xx = x + self.canvas.winfo_width()
        yy = y + self.canvas.winfo_height()
        ImageGrab.grab(bbox=(x, y, xx, yy)).save(fname)

    def toggle(self):
        self.playing.set(not self.playing.get())
        if self.playing.get():
            self.pauseB['text'] = 'Pause'
        else:
            self.pauseB['text'] = 'Play'

    def writeToFile(self):
        with open(os.path.join(self.dirname, 'coords.csv'), 'w') as f:
            w=csv.writer(f, quoting=csv.QUOTE_NONE)
            for i in range(self.n_particles):
                for j in range(len(self.coords[i][0])):
                    # data: frame #, particle #, x, y
                    w.writerow([int(j), i, self.coords[i][0][j], self.coords[i][1][j]])

    def run_simulation(self):
        # get coordinate information
        max_n_steps = 0
        i = 0
        n_exited = 0
        exit_times = []
        while i < self.n_particles:
            x_coords, y_coords, throw_out, exit_time, _, _, _, _ = move(
                self.total_time,
                self.pDriv,
                self.trap_dist,
                self.time_between,
                theta=2*np.pi*i/self.n_particles
            )
            max_n_steps = max(len(x_coords), max_n_steps)
            if throw_out:
                print(f"throw out particle {i}")
                continue
            else:
                if exit_time != -1:
                    n_exited += 1
                    exit_times.append(exit_time)
                self.coords.append([
                    [xd*self.scale_factor for xd in x_coords],
                    [yd*self.scale_factor for yd in y_coords]
                ])
                print(f"particle {i} done")
                i += 1

        print("{} out of {} exit the cell".format(n_exited, self.n_particles))
        print("Avg exit time: {:.2e}".format(np.mean(exit_times)))

        # change path as necessary
        img = tk.PhotoImage(file="dot-2.png")
        for i in range(self.n_particles):
            self.vsv.append(self.canvas.create_image(self.radius_x, self.radius_y, image=img))
            self.canvas.move(self.vsv[i], self.coords[i][0][0], self.coords[i][1][0])
        self.time_label.config(text='Time: {:7.2f} s'.format(0))
        self.canvas.update()

        if self.write_to_ps:
            self.playing.set(True)
            self.pauseB['text'] = 'PS'
            self.pauseB['state'] = 'disabled'
            self.resetB['state'] = 'disabled'
            self.csvB['state'] = 'disabled'
            self.csvB['text'] = 'Can\'t export to CSV'

        time.sleep(1)
        time.sleep(self.dt)

        while self.frame_number < max_n_steps:
            if not self.playing.get():
                self.pauseB.wait_variable(self.playing)
            for k in range(self.n_particles):
                try:
                    self.canvas.move(
                        self.vsv[k],
                        self.coords[k][0][self.frame_number] - self.coords[k][0][self.frame_number-1],
                        self.coords[k][1][self.frame_number] - self.coords[k][1][self.frame_number-1],
                    )
                except:
                    continue
            self.canvas.update()
            self.time_label.config(text='Time: {:7.2f} s'.format(self.frame_number*0.01))
            if self.write_to_ps:
                self.save_canvas(os.path.join(self.dirname, f"scene-{self.frame_number:03d}.tif"))
            else:
                time.sleep(self.dt)
            self.frame_number += 1
        if self.write_to_ps:
            self.root.destroy()

        self.root.mainloop()


if __name__ == "__main__":
    sim = Simulation(
        total_time=2000,
        n_particles=1,
        pDriv=0.03,
        trap_dist=TRAP_DIST,
        time_between=TIME_BETWEEN_STATES,
        dt=0.01,
        dirname="sim",
        width=600,
        height=600,
        write_to_ps=False,
    )
    sim.run_simulation()
