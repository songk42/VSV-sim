import numpy as np
import random
from matplotlib import pyplot as plt
from numba import njit, jit
import statistics as st
import tkinter as tk
import PIL.ImageGrab as ImageGrab
import time
import os
import csv

# @jit
def move(tTot, pDriv=0.03, ts=1.7e-7, avg=0.41, theta=0):
    '''tTot: maximum total amount of "cell time" this simulation is run
    pDriv: probability of driven motion as opposed to trap
    ts: distance between traps (m)
    avg: average time between states'''
    eta = 0.006 # Pa-s
    R = 8.6e-8 # m
    Kb = 1.38064852e-23 # Boltzmann constant
    T = 37 + 273.15 # K
    g = 6 * np.pi * eta * R
    k = random.uniform(6e-7, 2.6e-6) # spring constant; N/m
    D = Kb * T / g # diffusivity coefficient via Stokes-Einstein equation

    dt = 1e-2 # "framerate"

    # theta = 2 * np.pi * random.random()
    x = np.array([7.5e-6 * np.cos(theta)])
    y = np.array([7.5e-6 * np.sin(theta)])
    xC = x[0]
    yC = y[0]

    ts /= 2

    # effective probability, b/c pDriv should be time-based and this program doesn't treat it as such
    if pDriv != 0 and pDriv != 1:
        pDriv *= (avg / (1 - pDriv + avg*pDriv))

    # determine when particle changes states
    changeTime = np.array([], float)
    driven = {}
    sdf = 0.4
    inc = -1
    curr = random.random() < pDriv
    driven[0] = curr
    while inc < 0:
        if curr:
            inc = random.gauss(1, 1)
        else:
            inc = random.gauss(avg, avg)
    tempT = np.round(inc, 2)
    while tempT < tTot:
        changeTime = np.append(changeTime, tempT)
        curr = random.random() < pDriv
        driven[tempT] = curr
        inc = -1
        while inc < 0:
            if curr:
                inc = random.gauss(1, 1)
            else:
                inc = random.gauss(avg, avg)
        tempT = np.round(tempT + inc, 2)
        driven[np.round(tempT, 2)] = curr
    curr = driven[0]
    for t in np.arange(0, tTot+2*dt, dt):
        if t in changeTime:
            curr = driven[t]
        driven[np.round(t, 2)] = curr
        
    # changeTime = np.array([], float)
    # sdf = 0.4
    # inc = -1
    # while inc < 0:
    #     inc = random.gauss(avg, sdf)
    # tempT = np.round(inc, 2)
    # while tempT < tTot:
    #     changeTime = np.append(changeTime, tempT)
    #     inc = -1
    #     while inc < 0:
    #         inc = random.gauss(avg, sdf)
    #     tempT = np.round(tempT + inc, 2)
    # # print(changeTime)
    # # interstate = []
    # driven = {}
    # curr = random.random() < pDriv
    # for t in np.arange(0, tTot+2*dt, dt):
    #     driven[np.round(t, 2)] = curr
    #     if t in changeTime:
    #         curr = random.random() < pDriv

    exitTime = -1

    hop = np.array([], float)
    driv = np.array([], float)
    vd = np.array([], float)
    vh = np.array([], float)
    prev = 0

    adj = 0 # adjustment in change-state times due to failure to jump previously

    rev = False
    for t in np.arange(dt, tTot+dt, dt):
        t = np.round(t, 2)
        if driven[np.round(t, 2)]: # driven
            dr = random.gauss(2e-8, 2e-8)
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
            if np.sqrt(xtemp**2 + ytemp**2) >= 5e-6:
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
            if np.sqrt(xtemp**2 + ytemp**2) < 5e-6:
                return x, y, True, exitTime, hop, driv, vd, vh
            
            x = np.append(x, xtemp)
            y = np.append(y, ytemp)

            # change states from trap to something else
            if np.round(t-adj, 2) in changeTime:
                if not driven[np.round(t+dt, 2)]:
                    dr = -1
                    while dr < 0:
                        # sdf = 1.19
                        sdf = 1.8
                        dr = random.gauss(ts, sdf*ts) # idk what the SD should be
                    rand = 2
                    while np.abs(rand) > 1:
                        rand = random.gauss(0, 0.4)
                    theta = rand*2*np.pi
                    xC += dr * np.cos(theta)
                    yC += dr * np.sin(theta)

        # if np.round(t - dt - adj, 2) in changeTime:
        #     interstate.append(np.sqrt( (x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 ))

        # halve any differences (idk if this is a good idea, but it's a quick fix)
        x[-1] = (x[-1] + x[-2])/2
        y[-1] = (y[-1] + y[-2])/2

        # noise
        x[-1] += random.gauss(0, 2e-9)
        y[-1] += random.gauss(0, 2e-9)

        # determine the direction of the next bout of driven motion
        if np.round(t-adj, 2) in changeTime:
            rev = random.random()<0.5

        veltmp = np.sqrt( (x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 ) * 100

        # store velocity information
        if driven[np.round(t, 2)]:
            vd = np.append(vd, veltmp)
        else:
            vh = np.append(vh, veltmp)

        # distance traveled over the course of a particular state
        if np.round(t-adj, 2) in changeTime:
            tmp = np.sqrt((x[-1] - x[prev])**2 + (y[-1] - y[prev])**2)
            if driven[np.round(t, 2)]:
                driv = np.append(driv, tmp)
            else:
                hop = np.append(hop, tmp)
            prev = len(x)
        
        if np.sqrt(x[-1]**2+y[-1]**2) > 1.502e-5 and exitTime == -1:
            exitTime = t
            # break # uncomment this line when running the graphics part of simulation

    # fi.close()
    # print(interstate)
    return x, y, False, exitTime, hop, driv, vd, vh



def graph(tTot):
    x, y = move(tTot)
    dt = 0.01
    gx = [i * 1e6 for i in x]
    gy = [i * 1e6 for i in y]
    for i in range(tTot-1):
        plt.scatter(gx[int(i/dt):int((i+1)/dt)], gy[int(i/dt):int((i+1)/dt)])
    centerx = []
    centery = []
    for i in range(tTot-1):
        centerx.append(st.mean(gx[int(i/dt):int((i+1)/dt)]))
        centery.append(st.mean(gy[int(i/dt):int((i+1)/dt)]))
    plt.plot(centerx, centery)
    plt.show()
    
def run_graphics(tTot, n=1, psW=False, pDriv=0.03, ts=1.7e-7, avg=0.41, dirname="sim"):
    '''tTot: maximum total amount of "cell time" this simulation is run
    n: number of particles shown
    psW: if true, write canvas as Postscript files
    pDriv: probability of driven motion as opposed to trap
    ts: size of trap (m)
    avg: average time between states'''
    root = tk.Tk()

    WIDTH = 1600
    HEIGHT = 1600

    CELL = 1.502e-5
    NUC = 5e-6
 
    # change directory as needed
    dir = "/home/skim52/RSI/sim-data/{}/".format(dirname)
    try:
        os.mkdir(dir)
    except:
        pass

    x1 = WIDTH / 2
    y1 = HEIGHT / 2

    sc = 5e7

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
    canvas.grid(row=0, columnspan=3)
    canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill='black')
    # cell, nucleus boundaries
    canvas.create_oval(x1-CELL*sc, y1-CELL*sc, x1+CELL*sc, y1+CELL*sc, outline='yellow')
    canvas.create_oval(x1-NUC*sc, y1-NUC*sc, x1+NUC*sc, y1+NUC*sc, outline='white')

    timeL = tk.Label(root, font=('arial', '30'))
    timeL.grid(row=1, column=1)

    def save_canvas(fname):
        nonlocal root
        nonlocal canvas
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        xx = x + canvas.winfo_width()
        yy = y + canvas.winfo_height()
        ImageGrab.grab(bbox=(x, y, xx, yy)).save(fname)

    # get coordinate information
    coords = []
    dur = 0
    i = 0
    nout = 0
    exitTime = []
    while i < n:
        x, y, b, et, out0, out1, out2, out3 = move(tTot, pDriv, ts, avg, theta=2*np.pi*i/n)
        dur = max(len(x), dur)
        if b:
            i -= 1
            print("throw out")
        else:
            if et != -1:
                nout += 1
                exitTime.append(et)
            coords.append([[xd*sc for xd in x], [yd*sc for yd in y]])
            print(i)
        i += 1

    print("{} out of {} exit the cell".format(nout, n))
    print("Avg exit time: {:.2e}".format(np.mean(exitTime)))

    bot = tk.Frame()
    bot.grid(row=2, columnspan=3)

    pausetxt = tk.StringVar()
    pausetxt.set("Play")
    playBool = tk.BooleanVar()
    playBool.set(False)
    def toggle():
        nonlocal playBool
        playBool.set(not playBool.get())
        if pauseB['text'] == 'Play':
            pauseB['text'] = 'Pause'
        else:
            pauseB['text'] = 'Play'
    pauseB = tk.Button(bot, text='Play', command=toggle, font=('arial', '30'))
    pauseB.grid(row=0, column=0)

    r = 5
    dt = 0.01

    vsv = []
    # change path as necessary
    img = tk.PhotoImage(file="/home/skim52/RSI/code/dot-2.png")
    for i in range(n):
        vsv.append(canvas.create_image(x1, y1, image=img))
        canvas.move(vsv[i], coords[i][0][0], coords[i][1][0])
    timeL.config(text='Time: {:7.2f} s'.format(0))
    canvas.update()

    t = 1
    def reset():
        nonlocal canvas
        nonlocal coords
        nonlocal x1
        nonlocal y1
        nonlocal vsv
        nonlocal t
        for i in range(len(vsv)):
            c = canvas.coords(vsv[i])
            canvas.move(vsv[i], x1 + coords[i][0][0] - c[0], y1 + coords[i][1][0] - c[1])
        canvas.update()
        timeL.config(text='Time: {:7.2f} s'.format(0))
        t = 1
    resetB = tk.Button(bot, text='Reset', command=reset, font=('arial', '30'))
    resetB.grid(row=0, column=1)

    def writeToFile():
        nonlocal coords
        with open('coords.csv', 'w') as f:
            w=csv.writer(f, quoting=csv.QUOTE_NONE)
            for i in range(n):
                for j in range(len(coords[i][0])):
                    # data: frame #, particle #, x, y
                    w.writerow([int(j), i, coords[i][0][j], coords[i][1][j]])
    csvB = tk.Button(bot, text='Export to CSV', command=writeToFile, font=('arial', '30'))
    csvB.grid(row=0, column=2)
    if psW:
        playBool.set(True)
        pauseB['text'] = 'PS'
        pauseB['state'] = 'disabled'
        resetB['state'] = 'disabled'
        csvB['state'] = 'disabled'
        csvB['text'] = 'Can\'t export to CSV'

    time.sleep(1)
    time.sleep(dt)

    while t < dur:
        if not playBool.get():
            pauseB.wait_variable(playBool)
        for k in range(n):
            try:
                canvas.move(vsv[k], coords[k][0][t] - coords[k][0][t-1], coords[k][1][t] - coords[k][1][t-1])
            except:
                continue
        canvas.update()
        timeL.config(text='Time: {:7.2f} s'.format(t*0.01))
        if psW:
            # canvas.postscript(file="{}scene-{:03d}.ps".format(dir, t), colormode='color')
            save_canvas("{}scene-{:03d}.tif".format(dir, t))
        else:
            time.sleep(dt)
        t += 1
    if psW:
        root.destroy()
    
    root.mainloop()


# def analyze(tTot, n=1, pDriv=0.03, ts=1.7e-7, avg=0.41):
#     '''tTot: maximum total amount of "cell time" this simulation is run
#     n: number of particles shown
#     pDriv: probability of driven motion as opposed to trap
#     ts: size of trap (m)
#     avg: average time between states'''
#     coords = []
#     dur = 0
#     i = 0
#     nout = 0
#     exitTime = []
#     while i < n:
#         x, y, b, et = move(tTot, pDriv, ts, avg, theta=2*np.pi*i/n)
#         dur = max(len(x), dur)
#         if b:
#             i -= 1
#         else:
#             if et != -1:
#                 nout += 1
#                 exitTime.append(et)
#             coords.append([x, y])
#         i += 1

#     print("{} out of {} exit the cell".format(nout, n))
#     msd = []
#     for i in range(dur):
#         temp = []
#         for j in range(n):
#             try:
#                 temp.append(coords[j][0][i]**2 + coords[j][1][i]**2)
#             except:
#                 continue
#         msd.append(np.mean(temp))
#     taxis = [np.round(i/100, 2) for i in range(dur)]
#     return taxis, msd, exitTime


# run_graphics(4, 11)

def dispVsTime(pDriv):
    t = 1000
    gx=list(range(t))
    gy=[7.5]
    n=100
    tmp = [[] for i in range(t-1)]

    f = open("coord{}".format(int(100*pDriv)), "w")

    for j in range(n):
        b = True
        while b:
            x, y, b, out0, out1, out2, out3, out4 = move(t, pDriv)
        print(len(x))
        for i in range(len(x)):
            f.write("{} {}\n".format(x[i], y[i]))
        f.write("---\n")
        for i in range(t-1):
            tmp[i].append((x[100*(i+1)-1]**2 + y[100*(i+1)-1]**2) ** 0.5)

    f.close()

    for l in tmp:
        gy.append(np.mean(l) * 1e6)

    gy = [k - 7.5 for k in gy]
    print(gy)
    plt.ylim(0, 8)
    plt.xlabel("Time (s)")
    plt.ylabel("Dist traveled (um)")
    plt.scatter(gx, gy)
    plt.show()