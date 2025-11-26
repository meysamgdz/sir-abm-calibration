"""
Summary Statistics for ABC Calibration

Defines summary statistics that capture key features of epidemic curves.
These are used to compare observed vs simulated epidemics in ABC.

Key principle: Good summary statistics are:
- Low-dimensional (few values)
- Informative (capture important features)
- Robust (not sensitive to noise)
"""

import numpy as np
from scipy import stats


def calculate_summary_statistics(sir_data, num_agents=100):
    """
    Calculate comprehensive summary statistics from SIR time series.

    Args:
        sir_data: Either DataFrame with S,I,R columns or list of (S,I,R) tuples
        num_agents (int): Total population for normalization

    Returns:
        dict: Dictionary of summary statistics
    """
    # Handle different input formats
    if hasattr(sir_data, 'columns'):  # DataFrame
        s_curve = sir_data['S'].values
        i_curve = sir_data['I'].values
        r_curve = sir_data['R'].values
    else:  # List of tuples
        s_curve = np.array([x[0] for x in sir_data])
        i_curve = np.array([x[1] for x in sir_data])
        r_curve = np.array([x[2] for x in sir_data])

    # ========================================================================
    # Peak Statistics
    # ========================================================================

    peak_infected = np.max(i_curve)
    peak_time = np.argmax(i_curve)
    peak_prevalence = peak_infected / num_agents

    # ========================================================================
    # Attack Rate (Final Size)
    # ========================================================================

    final_recovered = r_curve[-1]
    attack_rate = final_recovered / num_agents

    # ========================================================================
    # Duration Statistics
    # ========================================================================

    # Time to peak
    time_to_peak = peak_time

    # Duration (when infections drop below threshold)
    threshold = max(1, 0.01 * num_agents)  # 1% or 1 person
    duration = len(i_curve)
    for t, infected in enumerate(i_curve):
        if infected < threshold and t > peak_time:
            duration = t
            break

    # ========================================================================
    # Growth Rate (Early Exponential Phase)
    # ========================================================================

    # Use first 10 days or until peak (whichever is smaller)
    growth_phase_length = min(10, peak_time)

    if growth_phase_length > 3:
        early_i = i_curve[1:growth_phase_length + 1]
        early_i = np.maximum(early_i, 1)  # Avoid log(0)

        # Exponential growth rate from log-linear fit
        x = np.arange(len(early_i))
        log_i = np.log(early_i)

        if len(x) > 1:
            growth_rate = np.polyfit(x, log_i, 1)[0]
        else:
            growth_rate = 0.0
    else:
        growth_rate = 0.0

    # ========================================================================
    # Curve Shape Statistics
    # ========================================================================

    # Skewness of infection curve
    if len(i_curve) > 3:
        curve_skewness = stats.skew(i_curve)
    else:
        curve_skewness = 0.0

    # Width of epidemic (time above half-maximum)
    half_max = peak_infected / 2
    above_half_max = np.sum(i_curve > half_max)

    # Area under infected curve (total infection-days)
    total_infection_days = np.sum(i_curve)

    # ========================================================================
    # Return Summary Statistics
    # ========================================================================

    summary = {
        # Peak statistics
        'peak_infected': float(peak_infected),
        'peak_time': float(peak_time),
        'peak_prevalence': float(peak_prevalence),

        # Final size
        'attack_rate': float(attack_rate),
        'final_susceptible': float(s_curve[-1]),

        # Temporal dynamics
        'time_to_peak': float(time_to_peak),
        'duration': float(duration),
        'growth_rate': float(growth_rate),

        # Shape
        'curve_skewness': float(curve_skewness),
        'width_half_max': float(above_half_max),
        'total_infection_days': float(total_infection_days),
    }

    return summary


def calculate_core_statistics(sir_data, num_agents=100):
    """
    Calculate minimal set of core statistics (for faster ABC).

    These 4 statistics capture the most important epidemic features:
    1. Peak time (when does it peak?)
    2. Peak infected (how big is the peak?)
    3. Attack rate (how many ultimately infected?)
    4. Growth rate (how fast does it spread?)

    Args:
        sir_data: SIR time series data
        num_agents (int): Total population

    Returns:
        np.ndarray: Array of 4 core statistics
    """
    full_stats = calculate_summary_statistics(sir_data, num_agents)

    core = np.array([
        full_stats['peak_time'],
        full_stats['peak_infected'],
        full_stats['attack_rate'],
        full_stats['growth_rate']
    ])

    return core


def euclidean_distance(stats1, stats2, weights=None):
    """
    Calculate weighted Euclidean distance between summary statistics.

    Args:
        stats1 (dict or array): First set of statistics
        stats2 (dict or array): Second set of statistics
        weights (dict or array): Weights for each statistic

    Returns:
        float: Distance measure
    """
    # Convert dicts to arrays if needed
    if isinstance(stats1, dict):
        keys = sorted(stats1.keys())
        vec1 = np.array([stats1[k] for k in keys])
        vec2 = np.array([stats2[k] for k in keys])

        if weights is not None and isinstance(weights, dict):
            w = np.array([weights.get(k, 1.0) for k in keys])
        else:
            w = np.ones(len(vec1))
    else:
        vec1 = np.array(stats1)
        vec2 = np.array(stats2)
        w = np.ones(len(vec1)) if weights is None else np.array(weights)

    # Normalize by standard deviation (if not zero)
    # This makes different-scale statistics comparable
    std = np.std([vec1, vec2], axis=0)
    std = np.where(std > 0, std, 1.0)  # Avoid division by zero

    normalized_diff = (vec1 - vec2) / std

    # Weighted Euclidean distance
    distance = np.sqrt(np.sum(w * normalized_diff ** 2))

    return distance


def compare_epidemics(observed_data, simulated_data, num_agents=100):
    """
    Compare two epidemic curves and return distance.

    Args:
        observed_data: Observed epidemic time series
        simulated_data: Simulated epidemic time series
        num_agents (int): Total population

    Returns:
        float: Distance between epidemics
    """
    # Use core statistics for comparison
    obs_core = calculate_core_statistics(observed_data, num_agents)
    sim_core = calculate_core_statistics(simulated_data, num_agents)

    distance = euclidean_distance(obs_core, sim_core)

    return distance


if __name__ == '__main__':
    # Test with dummy data
    print("Testing summary statistics...")

    # Create dummy epidemic curve
    days = 50
    peak_day = 20

    # Gaussian-like infection curve
    i_curve = 30 * np.exp(-0.5 * ((np.arange(days) - peak_day) / 5) ** 2)
    r_curve = np.cumsum(i_curve * 0.1)  # Cumulative recovered
    s_curve = 100 - i_curve - r_curve  # Susceptible

    # Create DataFrame-like structure
    import pandas as pd

    df = pd.DataFrame({
        'S': s_curve,
        'I': i_curve,
        'R': r_curve
    })

    # Calculate statistics
    stats = calculate_summary_statistics(df, num_agents=100)

    print("\nSummary Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key:25s}: {value:8.2f}")

    print("\n✅ Summary statistics module working!")