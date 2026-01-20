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
    mean_displacement: np.ndarray  # Mean displacement
    exit_times: np.ndarray  # Times at which particles exited the cell
    max_n_steps: int  # Maximum number of steps taken by any particle
    flux_trap: np.ndarray  # Flux for hopping motion
    flux_driven: np.ndarray  # Flux for driven motion
    distance_trap: np.ndarray  # Distance traveled during hopping
    distance_driven: np.ndarray  # Distance traveled during driven motion
    velocity_trap: np.ndarray  # Velocity during hopping
    velocity_driven: np.ndarray  # Velocity during driven motion


def analyze(config: sim.SimulationConfig, verbose: bool = False) -> AnalysisOutput:
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

    if verbose:
        it = tqdm.tqdm(range(config.n_particles), desc="Running simulations to analyze flux, distance, velocities, MSD")
    else:
        it = range(config.n_particles)

    for i in it:
        sim_output = sim.move(config, 2*np.pi*i/config.n_particles)
        max_n_steps = max(len(sim_output.x), max_n_steps)

        # Store flux
        fh = sum(sim_output.distance_trap) * len(sim_output.distance_trap)/max(1, len(sim_output.distance_trap)+len(sim_output.distance_driven)) # flux
        fd = sum(sim_output.distance_driven) * len(sim_output.distance_driven)/max(1, len(sim_output.distance_trap)+len(sim_output.distance_driven))
        flux_trap = np.append(flux_trap, fh/max(1, fd+fh))
        flux_driven = np.append(flux_driven, fd/max(1, fd+fh))

        # Store distances, velocities
        distance_trap = np.append(distance_trap, sum(sim_output.distance_trap))
        distance_driven = np.append(distance_driven, sum(sim_output.distance_driven))
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
    mean_displacement = np.zeros(max_n_steps)
    for x_traj, y_traj in zip(all_x_trajectories, all_y_trajectories):
        # Pad trajectories if needed to match max_n_steps
        if len(x_traj) < max_n_steps:
            x_traj = np.pad(x_traj, (0, max_n_steps - len(x_traj)), 'edge')
            y_traj = np.pad(y_traj, (0, max_n_steps - len(y_traj)), 'edge')
        dx = x_traj - x_traj[0]
        dy = y_traj - y_traj[0]
        msd += dx**2 + dy**2
        mean_displacement += np.sqrt(dx**2 + dy**2)

    msd /= config.n_particles
    mean_displacement /= config.n_particles
    if verbose:
        print(f"{nout} out of {config.n_particles} exit the cell".format(nout, config.n_particles))
        print(f"Mean exit time: {np.mean(exit_times)}")
        print(f"Mean distance (hop): {np.mean(distance_trap)}, (driv): {np.mean(distance_driven)}")
        print(f"Mean flux (hop): {np.mean(flux_trap)}, (driv): {np.mean(flux_driven)}")
        print(f"Mean velocity (hop): {np.mean(velocity_trap)}, (driv): {np.mean(velocity_driven)}")

    return AnalysisOutput(
        msd=msd,
        mean_displacement=mean_displacement,
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


def plot_displacement_vs_time(
    p_driv_vals,
    n_particles,
    total_time=600,
    dt=0.001,
    graph_cutoff=600,
):
    """
    Plot mean displacement vs time for multiple p_driv values.

    Parameters:
    -----------
    p_driv_vals : list of float
        List of p_driv values to test
    n_particles : int
        Number of particles to simulate for each p_driv
    total_time : float, optional
        Total simulation time in seconds (default: 600)
    colors : list of str, optional
        List of colors for each p_driv curve. If None, uses CSS4 colors
    dt : float, optional
        Time step in seconds (default: 0.001)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    for p_driv in p_driv_vals:
        print(f"\nRunning simulations for p_driv = {p_driv}")
        mean_displacement = np.zeros(int(total_time / dt + 2))
        particle_ct = np.zeros(int(total_time / dt + 2))
        max_len = 0
        for _ in tqdm.tqdm(range(n_particles)):
            config = sim.SimulationConfig(
                n_particles=1,
                total_time=total_time,
                p_driv=p_driv,
                end_early=False,
                dt=dt,
            )

            # Run analysis to get mean_displacement
            analysis_output = analyze(config)
            mean_displacement += np.pad(analysis_output.mean_displacement, (0, len(mean_displacement) - len(analysis_output.mean_displacement)), 'constant')
            particle_ct[:len(analysis_output.mean_displacement)] += 1
            max_len = max(max_len, len(analysis_output.mean_displacement))

        particle_ct[particle_ct == 0] = 1  # Prevent division by zero
        mean_displacement /= particle_ct
        mean_displacement = mean_displacement[:max_len]

        # Create time axis
        time = np.arange(len(mean_displacement)) * dt
        time_cutoff = min(int(graph_cutoff / dt), len(time))
        time = time[:time_cutoff]
        mean_displacement = mean_displacement[:time_cutoff]

        # Plot mean_displacement vs time
        ax.plot(time, mean_displacement * 1e6, label=f'p_driv = {p_driv}')
        del config, analysis_output

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(r'Mean Displacement ($\mu$m)', fontsize=12)
    ax.set_title('Mean Displacement vs Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax

import numpy as np

def compute_flux_4s(
        sim_output,
        cfg,
        window: float = 4.0,
        *,
        sample_dt: float = 0.01,
        rate: bool = True,
        return_mask: bool = False,
):
    """
    Returns per-4s distances (or rates) for diffusive and driven windows,
    plus a boolean mask indicating whether each window contains any driven motion.
    """
    import numpy as np

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

    t_orig = np.arange(len(sim_output.x)) * cfg.dt
    x_samp = np.interp(t_samples, t_orig, sim_output.x)
    y_samp = np.interp(t_samples, t_orig, sim_output.y)

    pts_per_win = int(round(window / sample_dt))
    n_windows   = (len(t_samples) - 1) // pts_per_win

    diff_flux  = np.zeros(n_windows, dtype=float)
    driv_flux  = np.zeros(n_windows, dtype=float)
    driven_msk = np.zeros(n_windows, dtype=bool)

    if not (hasattr(sim_output, "hop_t") and hasattr(sim_output, "hop_x") and hasattr(sim_output, "hop_y")):
        raise ValueError("SimulationOutput must include hop_t/hop_x/hop_y for trap-aware flux.")

    hop_t = np.asarray(sim_output.hop_t)
    hop_x = np.asarray(sim_output.hop_x)
    hop_y = np.asarray(sim_output.hop_y)

    for w in range(n_windows):
        s_idx = w * pts_per_win
        e_idx = s_idx + pts_per_win
        t_s   = t_samples[s_idx]
        t_e   = t_samples[e_idx]

        seg_state = state_samp[s_idx:e_idx]
        any_driv  = bool(seg_state.any())
        driven_msk[w] = any_driv

        if any_driv:
            # Treat "driven window" as end-to-end displacement over the window
            dx = x_samp[e_idx] - x_samp[s_idx]
            dy = y_samp[e_idx] - y_samp[s_idx]
            d_driv = float(np.hypot(dx, dy))
            d_diff = 0.0
        else:
            # Treat "diffusive window" as sum of hop center-to-center distances inside the window
            sel = np.where((hop_t > t_s) & (hop_t <= t_e))[0]
            d_driv = 0.0
            if sel.size < 2:
                d_diff = 0.0
            else:
                # pair consecutive hop centers within the window
                dx = np.diff(hop_x[sel])
                dy = np.diff(hop_y[sel])
                d_diff = float(np.sum(np.hypot(dx, dy)))

        if rate:  # convert distances to distance per window-second
            d_diff /= window
            d_driv /= window

        diff_flux[w] = d_diff
        driv_flux[w] = d_driv

    return (diff_flux, driv_flux, driven_msk) if return_mask else (diff_flux, driv_flux)


def plot_trap_size_histogram(
    trap_sizes,
    bins=30,
    robust=False,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    log: bool = False,
    winsorize: bool = False,
):
    """
    Plot histogram of trap sizes with optional robust outlier handling.

    Parameters:
    -----------
    trap_sizes : array-like
        Array of trap sizes (sigma) in meters
    bins : int or None, optional
        Number of bins for histogram
    robust : bool, optional
        If True, set x-range using percentile clipping to reduce outlier impact
    lower_pct : float, optional
        Lower percentile for robust range (default: 1.0)
    upper_pct : float, optional
        Upper percentile for robust range (default: 99.0)
    log : bool, optional
        If True, use log scale on x-axis
    winsorize : bool, optional
        If True, clip data to [lower_pct, upper_pct] before binning

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Convert to micrometers and clean invalid values
    trap_sizes_um = np.asarray(trap_sizes, dtype=float) * 1e6
    trap_sizes_um = trap_sizes_um[np.isfinite(trap_sizes_um)]

    fig, ax = plt.subplots(figsize=(6, 5))

    if trap_sizes_um.size == 0:
        ax.text(0.5, 0.5, "No trap sizes to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    data = trap_sizes_um.copy()

    # Determine robust limits
    if robust:
        lo = float(np.percentile(data, max(0.0, min(100.0, lower_pct))))
        hi = float(np.percentile(data, max(0.0, min(100.0, upper_pct))))
        if winsorize:
            data = np.clip(data, lo, hi)
        x_min, x_max = lo, hi
    else:
        x_min, x_max = float(np.min(data)), float(np.max(data))

    # Ensure positive min for log scale
    if log:
        if x_min <= 0:
            positives = data[data > 0]
            x_min = float(np.min(positives)) if positives.size else 1e-12

    ax.hist(data, bins=bins, density=False, edgecolor='black', color='skyblue')

    # Print statistics
    print(f"\nTrap Size Statistics:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: {np.mean(data):.4f} μm")
    print(f"  Median: {np.median(data):.4f} μm")
    print(f"  Std Dev: {np.std(data):.4f} μm")
    print(f"  Min: {np.min(data):.4f} μm")
    print(f"  Max: {np.max(data):.4f} μm")

    ax.set_xlabel(r'$\sigma$ ($\mu$m)', fontsize=12)
    ax.set_ylabel('count', fontsize=12)

    if log:
        ax.set_xscale('log')
    else:
        ax.set_xticks([0, 0.1, 0.2, 0.3])

    return fig, ax


def plot_trap_distance_histogram(sim_output, bins=30):
    """
    Plot histogram of distances between consecutive trap centers.

    Parameters:
    -----------
    sim_output : SimulationOutput
        Output from simulation containing hop_x and hop_y
    bins : int, optional
        Number of bins for histogram (default: 30)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Calculate distances between consecutive trap centers
    hop_x = sim_output.hop_x
    hop_y = sim_output.hop_y

    # Calculate center-to-center distances
    distances = []
    for i in range(1, len(hop_x)):
        dx = hop_x[i] - hop_x[i-1]
        dy = hop_y[i] - hop_y[i-1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)

    # Convert to micrometers
    distances_um = np.array(distances) * 1e6

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(distances_um, bins=bins, density=False, edgecolor='black', color='skyblue')

    # Print statistics
    print(f"\nTrap Distance Statistics:")
    print(f"  Count: {len(distances_um)}")
    print(f"  Mean: {np.mean(distances_um):.4f} μm")
    print(f"  Median: {np.median(distances_um):.4f} μm")
    print(f"  Std Dev: {np.std(distances_um):.4f} μm")
    print(f"  Min: {np.min(distances_um):.4f} μm")
    print(f"  Max: {np.max(distances_um):.4f} μm")

    ax.set_xlabel(r'center-to-center distance ($\mu$m)', fontsize=12)
    ax.set_ylabel('count', fontsize=12)
    ax.set_xlim(0, max(distances_um) * 1.2 if len(distances_um) > 0 else 1)
    ax.set_xticks([0, 0.5, 1, 1.5])

    return fig, ax


def compute_trap_sizes_from_simulation(sim_output, config):
    """
    Compute trap sizes (sigma) from simulation data by analyzing particle
    positions within each trap during diffusive periods.

    Parameters:
    -----------
    sim_output : SimulationOutput
        Simulation output containing trajectory and state data
    config : SimulationConfig
        Configuration with dt parameter

    Returns:
    --------
    trap_sizes : np.ndarray
        Array of trap sizes (standard deviations) in meters
    """
    hop_t = sim_output.hop_t
    hop_x = sim_output.hop_x
    hop_y = sim_output.hop_y
    state_dt = sim_output.state_dt
    x = sim_output.x
    y = sim_output.y

    trap_sizes = []

    # For each trap (except the last one), compute std dev of particle positions
    for i in range(len(hop_t) - 1):
        # Find time range for this trap
        t_start = hop_t[i]
        t_end = hop_t[i + 1]

        # Convert to indices
        idx_start = int(t_start / config.dt)
        idx_end = int(t_end / config.dt)

        # Only consider diffusive periods (not driven motion)
        trap_x_positions = []
        trap_y_positions = []

        for idx in range(idx_start, min(idx_end, len(x))):
            if idx < len(state_dt) and not state_dt[idx]:  # diffusive state
                trap_x_positions.append(x[idx])
                trap_y_positions.append(y[idx])

        # Fit 2D Gaussian by computing sigma_x and sigma_y, then average them
        if len(trap_x_positions) > 1:
            # Compute standard deviations in x and y directions
            sigma_x = np.std(trap_x_positions)
            sigma_y = np.std(trap_y_positions)

            # Average of sigma_x and sigma_y
            trap_size = (sigma_x + sigma_y) / 2
            trap_sizes.append(trap_size)

    trap_sizes = np.array(trap_sizes)
    trap_sizes = trap_sizes[trap_sizes < 0.1]
    return trap_sizes


def plot_msd_vs_time(
    p_driv_vals,
    n_particles,
    total_time=600,
    dt=0.001,
    graph_cutoff=600,
):
    """
    Plot mean squared displacement vs time for multiple p_driv values.

    Parameters:
    -----------
    p_driv_vals : list of float
        List of p_driv values to test
    n_particles : int
        Number of particles to simulate for each p_driv
    total_time : float, optional
        Total simulation time in seconds (default: 600)
    colors : list of str, optional
        List of colors for each p_driv curve. If None, uses CSS4 colors
    dt : float, optional
        Time step in seconds (default: 0.001)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    for p_driv in p_driv_vals:
        print(f"\nRunning simulations for p_driv = {p_driv}")
        msd = np.zeros(int(total_time / dt + 2))
        particle_ct = np.zeros(int(total_time / dt + 2))
        max_len = 0
        for _ in tqdm.tqdm(range(n_particles)):
            config = sim.SimulationConfig(
                n_particles=1,
                total_time=total_time,
                p_driv=p_driv,
                end_early=False,
                dt=dt,
            )

            # Run analysis to get MSD
            analysis_output = analyze(config)
            msd += np.pad(analysis_output.msd, (0, len(msd) - len(analysis_output.msd)), 'constant')
            particle_ct[:len(analysis_output.msd)] += 1
            max_len = max(max_len, len(analysis_output.msd))

        particle_ct[particle_ct == 0] = 1  # Prevent division by zero
        msd /= particle_ct
        msd = msd[:max_len]

        # Create time axis
        time = np.arange(len(msd)) * dt
        time_cutoff = min(int(graph_cutoff / dt), len(time))
        time = time[:time_cutoff]
        msd = msd[:time_cutoff]

        # Plot MSD vs time
        ax.plot(time, msd * 1e12, label=f'p_driv = {p_driv}')
        del config, analysis_output

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(r'Mean Squared Displacement ($\mu$m$^2$)', fontsize=12)
    ax.set_title('Mean Squared Displacement vs Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax

