import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import stats as st
import os
import simulation as sim
from numba import njit, jit
# import tkinter as tk
# import PIL.ImageGrab as ImageGrab
# import time
# import csv

def writefile(x, y, i, pDriv):
    f = open("/home/skim52/RSI/code/coord-test/coords-{}-{}".format(pDriv, i), 'w')
    for j in range(len(x)):
        f.write("{},{}\n".format(x[j], y[j]))
    f.close()

@jit
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
    taxis = [np.round(i/100, 2) for i in range(dur)]
    return taxis, msd, exitTime


# move(2000)

# run_graphics(10, 1, pDriv=0.5)

# sim8: driven stepsize = 2*D*dr whatever

# c = 0
# n = 500
# t = 300
# for i in range(n):
#     m = move(t, pDriv=0.06)
#     if len(m[0]) < t*100+1:
#         c += 1
#     if i % 10 == 0:
#         print(i)
# print("{} out of {} exit the cell after {} s".format(c, n, t))

time = 4000
n = 20

taxis, msd, et = analyze(time, n, 0.03)
# plt.plot(taxis, msd)
print(et)
print(np.mean(et), np.std(et))
print('---')
os.system('aplay /home/skim52/drip.wav')
taxis0, msd0, et0 = analyze(time, n, 0)
# plt.plot(taxis0, msd0)
print(et0)
print(np.mean(et0), np.std(et0))
print("---")
os.system('aplay /home/skim52/drip.wav')
# t, p = st.ttest_ind(et, et0)
# print("t = {}, p-val = {}".format(t, p))
# plt.title("MSD, P = 0 vs. 0.03")
# plt.xlabel("Time (s)")
# plt.ylabel("MSD")
# plt.show()


# msd stuff
# msd = [[] for i in range(200)]
# for i in range(1, 201):
#     for j in range(len(x)-1):
#         d2 = (x[j+i] - x[j])**2 + (y[j+i] - y[j])**2
#         msd[i-1].append(np.log10(d2))
# logmsd = [np.mean(l) for l in msd]
# logtau = [np.log10(i/100) for i in range(1,201)]
# plt.plot(logtau, logmsd)
# plt.show()


# p = 0.03, 1000s, driven step size = 1e-8
# 117 out of 227 exit the cell
# 7.290553015914073e-05 6.1562619214884255e-06
# 0.9970605354258748 0.0029394645741250523
# 7.653227169438374e-06 6.484809529593266e-07
# [675.79, 574.86, 776.74, 972.28, 317.56, 555.19, 515.24, 681.38, 389.02, 650.63, 935.99, 354.01, 988.48, 816.39, 666.08, 741.53, 565.57, 818.57, 563.19, 536.87, 372.95, 674.1, 335.98, 618.11, 576.12, 693.54, 578.18, 530.89, 460.08, 275.85, 583.25, 851.68, 513.2, 360.34, 584.13, 772.56, 854.14, 940.96, 777.06, 842.73, 710.18, 762.21, 694.93, 825.55, 381.81, 654.12, 946.13, 268.99, 559.87, 931.1, 548.94, 637.78, 890.71, 463.31, 690.28, 829.19, 942.43, 392.31, 943.65, 419.72, 823.98, 711.32, 903.74, 711.79, 591.31, 386.76, 802.28, 501.05, 716.61, 390.79, 953.03, 487.65, 542.64, 490.9, 948.0, 553.44, 547.79, 581.21, 656.97, 869.0, 537.16, 909.35, 776.4, 488.66, 809.29, 515.89, 180.31, 514.04, 617.72, 957.84, 399.75, 205.21, 646.43, 968.46, 494.19, 457.31, 541.21, 592.93, 524.67, 931.86, 845.33, 503.85, 747.67, 850.45, 922.18, 593.41, 880.81, 700.08, 672.13, 962.85, 742.32, 529.58, 290.74, 415.84, 393.47, 828.01, 798.45]
# 647.0131623931624 198.5604882612507
# ---
# ditto, p = 0
# 108 out of 227 exit the cell
# 7.172924541598569e-05 0.0
# 1.0 0.0
# /home/skim52/.local/lib/python3.6/site-packages/numpy/core/fromnumeric.py:3373: RuntimeWarning: Mean of empty slice.
#   out=out, **kwargs)
# /home/skim52/.local/lib/python3.6/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
#   ret = ret.dtype.type(ret / rcount)
# 7.657093084482338e-06 nan
# [537.99, 561.19, 900.8, 938.67, 569.99, 839.73, 619.13, 378.18, 373.42, 543.6, 555.56, 409.64, 457.06, 886.34, 719.47, 823.85, 777.09, 626.33, 817.1, 424.59, 851.61, 885.83, 899.29, 769.1, 742.58, 976.47, 623.06, 843.98, 722.69, 943.79, 681.03, 773.95, 975.25, 516.9, 482.65, 786.55, 442.07, 979.95, 774.39, 748.87, 491.42, 735.09, 734.39, 936.7, 925.78, 703.03, 475.28, 312.77, 749.71, 958.08, 668.52, 637.75, 312.67, 423.49, 676.21, 833.58, 977.36, 930.01, 730.15, 816.27, 725.84, 888.81, 857.4, 844.64, 308.52, 825.07, 818.54, 764.98, 870.76, 510.78, 672.86, 633.55, 582.93, 310.48, 965.4, 411.51, 681.92, 304.19, 376.62, 323.02, 578.56, 443.36, 635.2, 239.41, 726.39, 971.57, 156.53, 435.94, 482.96, 696.13, 584.66, 375.0, 725.2, 840.99, 659.23, 954.52, 937.86, 524.98, 341.12, 868.98, 503.0, 295.27, 995.84, 633.43, 517.54, 577.3, 589.52, 335.16]
# 662.1432407407409 210.19382729082287
# 
# Ttest_indResult(statistic=-0.5527163320937797, pvalue=0.5810112698880893)
# but also like they didn't all exit the cell

