from typing import NamedTuple

import numpy as np
import os

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import simulation as sim
import tqdm

def writefile(x, y, i, p_driv):
    if not os.path.exists("coord-test"):
        os.makedirs("coord-test")
    f = open("coord-test/coords-{}-{}".format(p_driv, i), 'w')
    for j in range(len(x)):
        f.write("{},{}\n".format(x[j], y[j]))
    f.close()


class AnalysisOutput(NamedTuple):
    msd: np.ndarray  # Mean squared displacement
    exit_times: np.ndarray  # Times at which particles exited the cell
    max_n_steps: int  # Maximum number of steps taken by any particle
    flux_trap: np.ndarray  # Flux for hopping motion
    flux_driven: np.ndarray  # Flux for driven motion
    distance_trap: np.ndarray  # Distance traveled during hopping
    distance_driven: np.ndarray  # Distance traveled during driven motion
    velocity_trap: np.ndarray  # Velocity during hopping
    velocity_driven: np.ndarray  # Velocity during driven motion


def analyze(config: sim.SimulationConfig):
    '''tTot: maximum total amount of "cell time" this simulation is run
    n: number of particles shown
    p_driv: probability of driven motion as opposed to trap
    trap_size: size of trap (m)
    avg: average time between states
    dt: time step (s)
    '''
    exit_times = np.array([])
    max_n_steps = 0
    i = 0
    nout = 0
    flux_trap = np.array([], float)
    flux_driven = np.array([], float)
    distance_trap = np.array([], float)
    distance_driven = np.array([], float)
    velocity_trap = np.array([], float)
    velocity_driven = np.array([], float)
    all_x_trajectories = []
    all_y_trajectories = []

    for i in tqdm.tqdm(range(config.n_particles), desc="Running simulations to analyze flux, distance, velocities, MSD"):
        sim_output = sim.move(config, 2*np.pi*i/config.n_particles)
        max_n_steps = max(len(sim_output.x), max_n_steps)

        # Store flux
        fh = sum(sim_output.distance_trap) * len(sim_output.distance_trap)/max(1, len(sim_output.distance_trap)+len(sim_output.distance_driven)) # flux
        fd = sum(sim_output.distance_driven) * len(sim_output.distance_driven)/max(1, len(sim_output.distance_trap)+len(sim_output.distance_driven))
        flux_trap = np.append(flux_trap, fh/max(1, fd+fh))
        flux_driven = np.append(flux_driven, fd/max(1, fd+fh))

        # Store distances, velocities
        distance_trap = np.append(distance_trap, sum(sim_output.velocity_trap))
        distance_driven = np.append(distance_driven, sum(sim_output.velocity_driven))
        velocity_trap = np.concatenate((velocity_trap, sim_output.velocity_trap))
        velocity_driven = np.concatenate((velocity_driven, sim_output.velocity_driven))

        # Collect trajectories for MSD calculation
        all_x_trajectories.append(np.array(sim_output.x))
        all_y_trajectories.append(np.array(sim_output.y))

        if sim_output.exit_time != -1:
            nout += 1
            exit_times = np.append(exit_times, sim_output.exit_time)

        # Collect trajectories for MSD calculation
        all_x_trajectories.append(np.array(sim_output.x))
        all_y_trajectories.append(np.array(sim_output.y))


    msd = np.zeros(max_n_steps)
    for x_traj, y_traj in zip(all_x_trajectories, all_y_trajectories):
        # Pad trajectories if needed to match max_n_steps
        if len(x_traj) < max_n_steps:
            x_traj = np.pad(x_traj, (0, max_n_steps - len(x_traj)), 'edge')
            y_traj = np.pad(y_traj, (0, max_n_steps - len(y_traj)), 'edge')
        dx = x_traj - x_traj[0]
        dy = y_traj - y_traj[0]
        msd += dx**2 + dy**2

    msd /= max_n_steps
    print(f"{nout} out of {config.n_particles} exit the cell".format(nout, config.n_particles))
    print(f"Mean exit time: {np.mean(exit_times)}")
    print(f"Mean distance (hop): {np.mean(distance_trap)}, (driv): {np.mean(distance_driven)}")
    print(f"Mean flux (hop): {np.mean(flux_trap)}, (driv): {np.mean(flux_driven)}")
    print(f"Mean velocity (hop): {np.mean(velocity_trap)}, (driv): {np.mean(velocity_driven)}")

    return AnalysisOutput(
        msd=msd,
        exit_times=exit_times,
        max_n_steps=max_n_steps,
        flux_trap=flux_trap,
        flux_driven=flux_driven,
        distance_trap=distance_trap,
        distance_driven=distance_driven,
        velocity_trap=velocity_trap,
        velocity_driven=velocity_driven,
    )


def displacement_vs_time(
    total_time,
    n_particles,
    p_driv,
):
    x_all = []
    y_all = []
    config = sim.SimulationConfig(
        n_particles=n_particles,
        total_time=total_time,
        p_driv=p_driv,
        end_early=False,
    )

    for _ in tqdm.tqdm(range(n_particles), desc="Running simulations for displacement vs time"):
        sim_output = sim.move(config)
        x = np.concatenate([np.array([7.5]), sim_output.x * 1e6])  # Convert to micrometers
        y = np.concatenate([np.array([0]), sim_output.y * 1e6])
        x_all.append(x)
        y_all.append(y)

    # Find the maximum trajectory length
    max_len = max(len(x) for x in x_all)

    # Pad each trajectory with the last value to make them all the same length
    x_all_padded = [np.pad(x, (0, max_len - len(x)), mode='edge') for x in x_all]
    y_all_padded = [np.pad(y, (0, max_len - len(y)), mode='edge') for y in y_all]

    x_all = np.array(x_all_padded)
    y_all = np.array(y_all_padded)
    displacements = np.sqrt(x_all**2 + y_all**2)
    displacements -= 7.5 # Center the displacement around 0
    mean_dist = np.mean(displacements, axis=0)

    displacements = np.abs(displacements)
    mean_disp = np.mean(displacements, axis=0)
    mean_squared_disp = np.mean(displacements**2, axis=0)

    return mean_dist, mean_disp, mean_squared_disp


def plot_displacement_vs_time_line(
        p_driv_vals, # list of percentages of driven motion which should be tested
        n_particles,
        colors = None,
        total_time = 600,
        dt = 0.001
):
    if colors is None:
        colors = list(mcolors.CSS4_COLORS.keys())[:len(p_driv_vals)]
    with tqdm.tqdm(total=len(p_driv_vals) * n_particles, desc="Running simulations for each driven probability", unit="particle") as pbar:
        for i, p_driv in enumerate(p_driv_vals):
            config = sim.SimulationConfig(
                n_particles=n_particles,
                total_time=total_time,
                p_driv=p_driv,
                end_early=False,
            )
            # print(f"Running move for p_driv {config.p_driv}")

            x_all = []
            y_all = []
            for _ in range(n_particles):
                sim_output = sim.move(config, stop_on_cell_exit=False)
                x = np.concatenate([np.array([7.5]), sim_output.x * 1e6])  # Convert to micrometers
                y = np.concatenate([np.array([0]), sim_output.y * 1e6])
                x_all.append(x)
                y_all.append(y)
                pbar.update(1)

            # Find the maximum trajectory length
            max_len = max(len(x) for x in x_all)

            # Pad each trajectory with the last value to make them all the same length
            x_all_padded = np.array([np.pad(x, (0, max_len - len(x)), mode='edge') for x in x_all])
            y_all_padded = np.array([np.pad(y, (0, max_len - len(y)), mode='edge') for y in y_all])

            # Calculate mean signed displacement around 0
            displacements = np.sqrt(x_all_padded**2 + y_all_padded**2)
            displacements -= 7.5 # Center the displacement around 0
            mean_signed = np.mean(displacements, axis=0)

            # Set up time axis
            t = np.arange(len(mean_signed)) * dt    # time axis (s)

            # Plot
            plt.plot(t, mean_signed, label=p_driv, color=colors[i])

import numpy as np

def compute_flux_4s(
        sim_output,
        cfg,
        window: float = 4.0,
        *,
        sample_dt: float = 0.01,
        rate: bool = True,
        return_mask: bool = False,
        fraction_weighting: bool = True,
        driven_fraction: float = 0.0368,  # 3.68% time in driven state
):
    """
    Trap-aware 4-s distances/rates with optional paper-style fraction weighting.

    - DRIVEN window (any driven frames): driven = end-to-end over window; diffusive = 0
    - DIFFUSIVE-only window: diffusive = sum of center-to-center hop distances whose
      hop times fall inside the window; driven = 0
    - If fraction_weighting=True: multiply distances for each state by the global
      fraction of time in that state (driven_fraction, 1-driven_fraction) to match paper.
    """
    import numpy as np

    # --- 0.01 s sampling grid + states (reuse your existing logic) ---
    t_end = (len(sim_output.x) - 1) * cfg.dt
    if hasattr(sim_output, "t_01s") and hasattr(sim_output, "state_01s") and abs(sample_dt - 0.01) < 1e-12:
        t_samples  = sim_output.t_01s
        state_samp = sim_output.state_01s.astype(bool)
        keep = t_samples <= t_end + 1e-12
        t_samples  = t_samples[keep]
        state_samp = state_samp[keep]
    else:
        t_samples = np.round(np.arange(0.0, t_end + 1e-12, sample_dt), 2)
        idx = np.minimum((t_samples / cfg.dt).astype(np.int64), len(sim_output.state_dt) - 1)
        state_samp = sim_output.state_dt[idx].astype(bool)

    # sample positions at same grid
    t_orig = np.arange(len(sim_output.x)) * cfg.dt
    x_samp = np.interp(t_samples, t_orig, sim_output.x)
    y_samp = np.interp(t_samples, t_orig, sim_output.y)

    # --- windows by index and by time ---
    pts_per_win = int(round(window / sample_dt))
    n_windows   = (len(t_samples) - 1) // pts_per_win  # need at least one step per window

    diff_flux  = np.zeros(n_windows, dtype=float)
    driv_flux  = np.zeros(n_windows, dtype=float)
    driven_msk = np.zeros(n_windows, dtype=bool)

    # hop event data (times in seconds, centers in meters)
    if not (hasattr(sim_output, "hop_t") and hasattr(sim_output, "hop_x") and hasattr(sim_output, "hop_y")):
        raise ValueError("SimulationOutput must include hop_t/hop_x/hop_y for trap-aware flux.")

    hop_t = np.asarray(sim_output.hop_t)
    hop_x = np.asarray(sim_output.hop_x)
    hop_y = np.asarray(sim_output.hop_y)

    diff_weight  = 1.0 - driven_fraction
    driv_weight  = driven_fraction

    for w in range(n_windows):
        s_idx = w * pts_per_win
        e_idx = s_idx + pts_per_win
        t_s   = t_samples[s_idx]
        t_e   = t_samples[e_idx]

        seg_state = state_samp[s_idx:e_idx]
        any_driv  = bool(seg_state.any())
        driven_msk[w] = any_driv

        if any_driv:
            # driven = end-to-end over the window (Holzwarth-style)
            dx = x_samp[e_idx] - x_samp[s_idx]
            dy = y_samp[e_idx] - y_samp[s_idx]
            d_driv = float(np.hypot(dx, dy))
            d_diff = 0.0
        else:
            # diffusive = sum of center-to-center distances for hops within (t_s, t_e]
            sel = np.where((hop_t > t_s) & (hop_t <= t_e))[0]
            d_driv = 0.0
            if sel.size == 0:
                d_diff = 0.0
            else:
                j_prev = sel - 1
                j_prev = j_prev[j_prev >= 0]
                sel    = sel[:len(j_prev)]
                dx = hop_x[sel] - hop_x[j_prev]
                dy = hop_y[sel] - hop_y[j_prev]
                d_diff = float(np.sum(np.hypot(dx, dy)))

        # convert to rate if requested
        if rate:
            d_diff /= window
            d_driv /= window

        # Apple weighting per state --y
        if fraction_weighting:
            d_diff *= diff_weight
            d_driv *= driv_weight

        diff_flux[w] = d_diff
        driv_flux[w] = d_driv

    return (diff_flux, driv_flux, driven_msk) if return_mask else (diff_flux, driv_flux)


