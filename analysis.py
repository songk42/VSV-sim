import numpy as np
from matplotlib import pyplot as plt
import os
import simulation as sim

def writefile(x, y, i, pDriv):
    if not os.path.exists("coord-test"):
        os.makedirs("coord-test")
    f = open("coord-test/coords-{}-{}".format(pDriv, i), 'w')
    for j in range(len(x)):
        f.write("{},{}\n".format(x[j], y[j]))
    f.close()

def analyze(
    tTot,
    n=1,
    pDriv=0.03,
    trap_size=sim.TRAP_SIZE,
    trap_std=sim.TRAP_STD,
    avg=0.41,
    dt=0.001,
):
    '''tTot: maximum total amount of "cell time" this simulation is run
    n: number of particles shown
    pDriv: probability of driven motion as opposed to trap
    trap_size: size of trap (m)
    avg: average time between states
    dt: time step (s)
    '''
    # coords = []
    exitTime = np.array([])
    max_n_steps = 0
    i = 0
    nout = 0
    fhop = np.array([], float)
    fdriv = np.array([], float)
    dhop = np.array([], float)
    ddriv = np.array([], float)
    vhop = np.array([], float)
    vdriv = np.array([], float)
    for i in range(n):
        x, y, et, hop, driv, vd, vh = sim.move(tTot, pDriv, trap_size, trap_std, avg, 2*np.pi*i/n, dt)
        writefile(x, y, i, pDriv)
        max_n_steps = max(len(x), max_n_steps)
        fh = sum(hop) * len(hop)/max(1, len(hop)+len(driv)) # flux
        fd = sum(driv) * len(driv)/max(1, len(hop)+len(driv))
        fhop = np.append(fhop, fh/max(1, fd+fh))
        fdriv = np.append(fdriv, fd/max(1, fd+fh))
        dhop = np.append(dhop, sum(hop))
        ddriv = np.append(ddriv, sum(driv))
        for j in vh:
            vhop = np.append(vhop,j)
        for j in vd:
            vdriv = np.append(vdriv,j)
        if et != -1:
            nout += 1
            exitTime = np.append(exitTime, et)

    # print("---")

    print(f"{nout} out of {n} exit the cell".format(nout, n))
    print(f"Mean exit time: {np.mean(exitTime)}")
    print(f"Mean distance (hop): {np.mean(dhop)}, (driv): {np.mean(ddriv)}")
    print(f"Mean flux (hop): {np.mean(fhop)}, (driv): {np.mean(fdriv)}")
    print(f"Mean velocity (hop): {np.mean(vhop)}, (driv): {np.mean(vdriv)}")
    msd = []
    # for i in range(max_n_steps):
    #     temp = []
    #     for j in range(n):
    #         try:
    #             temp.append(coords[j][0][i]**2 + coords[j][1][i]**2)
    #         except:
    #             continue
    #     msd.append(np.mean(temp))
    time_axis = dt * np.arange(max_n_steps)
    return time_axis, msd, exitTime


def dispVsTime(pDriv):
    t = 1000
    gx=list(range(t))
    gy=[7.5]
    n=100
    tmp = [[] for i in range(t-1)]

    f = open("coord{}".format(int(100*pDriv)), "w")

    for _ in range(n):
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
