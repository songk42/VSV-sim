import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import stats as st
import os
import simulation as sim
# import tkinter as tk
# import PIL.ImageGrab as ImageGrab
# import time
# import csv

def writefile(x, y, i, pDriv):
    if not os.path.exists("coord-test"):
        os.makedirs("coord-test")
    f = open("coord-test/coords-{}-{}".format(pDriv, i), 'w')
    for j in range(len(x)):
        f.write("{},{}\n".format(x[j], y[j]))
    f.close()

def analyze(tTot, n=1, pDriv=0.03, ts=1.7e-7, avg=0.41):
    '''tTot: maximum total amount of "cell time" this simulation is run
    n: number of particles shown
    pDriv: probability of driven motion as opposed to trap
    ts: size of trap (m)
    avg: average time between states'''
    # coords = []
    exitTime = np.array([])
    dur = 0
    i = 0
    nout = 0
    fhop = np.array([], float)
    fdriv = np.array([], float)
    dhop = np.array([], float)
    ddriv = np.array([], float)
    vhop = np.array([], float)
    vdriv = np.array([], float)
    while i < n:
        x, y, b, et, hop, driv, vd, vh = sim.move(tTot, pDriv, ts, avg, theta=2*np.pi*i/n)
        writefile(x, y, i, pDriv)
        # sum(hop) = tot. distance achieved by one particle via hopping
        dur = max(len(x), dur)
        fh = sum(hop) * len(hop)/(len(hop)+len(driv)) # flux
        fd = sum(driv) * len(driv)/(len(hop)+len(driv))
        # print(len(hop), len(driv))
        fhop = np.append(fhop, fh/(fd+fh))
        fdriv = np.append(fdriv, fd/(fd+fh))
        dhop = np.append(dhop, sum(hop))
        ddriv = np.append(ddriv, sum(driv))
        for j in vh:
            vhop = np.append(vhop,j)
        for j in vd:
            vdriv = np.append(vdriv,j)
        if b:
            i -= 1
            # print("throw out")
        else:
            if et != -1:
                nout += 1
                exitTime = np.append(exitTime, et)
            # coords.append([x, y])
            # print(i)
            if i % 50 == 0:
                print(i)
        i += 1

    # print("---")

    print("{} out of {} exit the cell".format(nout, n))
    print(np.mean(dhop), np.mean(ddriv)) # mean distance via each method
    print(np.mean(fhop), np.mean(fdriv)) # mean fraction of flux via each method???
    print(np.mean(vhop), np.mean(vdriv)) # mean velocity via each method?
    msd = []
    # for i in range(dur):
    #     temp = []
    #     for j in range(n):
    #         try:
    #             temp.append(coords[j][0][i]**2 + coords[j][1][i]**2)
    #         except:
    #             continue
    #     msd.append(np.mean(temp))
    time_axis = [np.round(i/100, 2) for i in range(dur)]
    return time_axis, msd, exitTime


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
            x, y, b, _, _, _, _, out4 = sim.move(t, pDriv)
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
