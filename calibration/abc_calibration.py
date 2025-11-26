"""
ABC Calibration with PyMC

Implements Approximate Bayesian Computation (ABC) to calibrate SIR model
parameters (beta and gamma) using observed epidemic data.

Uses PyMC's ABC-SMC (Sequential Monte Carlo) implementation for efficient
parameter inference.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import sys
import os
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import run_sir_simulation
from calibration.summary_stats import calculate_core_statistics


def sir_simulator(beta, gamma, num_agents=100, num_infected=1,
                  max_steps=200):
    """
    Wrapper for SIR simulation that returns summary statistics.

    This function is called repeatedly by ABC to generate simulated data
    for different parameter values.

    Args:
        beta (float): Transmission rate
        gamma (float): Recovery rate
        num_agents (int): Population size
        num_infected (int): Initial infected
        max_steps (int): Maximum simulation steps
        population_size (int): Population for normalization

    Returns:
        np.ndarray: Summary statistics [peak_time, peak_infected, attack_rate, growth_rate]
    """
    # Ensure valid parameters
    beta = float(np.clip(beta, 0.001, 0.999))
    gamma = float(np.clip(gamma, 0.001, 0.999))

    infectious_period = 1.0 / gamma

    try:
        # Run simulation
        history, _, _ = run_sir_simulation(
            num_agents=num_agents,
            num_infected=num_infected,
            beta=beta,
            gamma=gamma,
            infectious_period=infectious_period,
            max_steps=max_steps,
            seed=None  # Random seed for stochasticity
        )

        # Calculate summary statistics
        summary_stats = calculate_core_statistics(sir_data=history, num_agents=num_agents)

        return summary_stats

    except Exception as e:
        print(f"Simulation failed for beta={beta}, gamma={gamma}: {e}")
        # Return extreme values to ensure rejection
        return np.array([999.0, 0.0, 0.0, 0.0])


def load_observed_data(data_path='data/observed_epidemic.csv'):
    """
    Load observed epidemic data.

    Args:
        data_path (str): Path to observed data CSV

    Returns:
        tuple: (DataFrame, summary_statistics)
    """
    df = pd.read_csv(data_path)

    # Calculate summary statistics from observed data
    obs_stats = calculate_core_statistics(sir_data=df, num_agents=len(df))

    print(f"Loaded observed data from: {data_path}")
    print(f"  Duration: {len(df)} days")
    print(f"  Peak infected: {df['I'].max()}")
    print(f"  Final recovered: {df['R'].iloc[-1]}")

    return df, obs_stats


def run_abc_calibration(observed_stats,
                        num_agents=100,
                        num_infected=1,
                        n_samples=2000,
                        n_steps=4,
                        epsilon=None,
                        output_dir='results'):
    """
    Run ABC-SMC calibration to infer beta and gamma.

    Args:
        observed_stats (np.ndarray): Observed summary statistics
        num_agents (int): Population size for simulations
        num_infected (int): Initial infected for simulations
        n_samples (int): Number of samples per SMC iteration
        n_steps (int): Number of SMC iterations (more = better convergence)
        epsilon (float): Distance threshold (if None, uses adaptive)
        output_dir (str): Directory to save results

    Returns:
        tuple: (trace, model) - PyMC trace and model objects
    """
    print("\n" + "=" * 70)
    print("ABC-SMC CALIBRATION")
    print("=" * 70)

    print(f"\nObserved summary statistics:")
    print(f"  Peak time: {observed_stats[0]:.1f}")
    print(f"  Peak infected: {observed_stats[1]:.1f}")
    print(f"  Attack rate: {observed_stats[2]:.3f}")
    print(f"  Growth rate: {observed_stats[3]:.3f}")

    print(f"\nABC Settings:")
    print(f"  Samples per iteration: {n_samples}")
    print(f"  SMC steps: {n_steps}")
    print(f"  Epsilon: {'adaptive' if epsilon is None else epsilon}")

    # Create PyMC model
    with pm.Model() as model:
        # Priors for beta and gamma
        # Uniform priors (uninformative)
        beta = pm.Uniform('beta', lower=0.01, upper=0.8)
        gamma = pm.Uniform('gamma', lower=0.05, upper=0.5)

        # Simulator
        # Note: PyMC's Simulator is designed for ABC
        sim = pm.Simulator(
            'sim',
            sir_simulator,
            params=(beta, gamma),
            epsilon=epsilon,
            observed=observed_stats,
            distance='euclidean',
            sum_stat='identity',  # We already computed summary stats
            ndim_supp=len(observed_stats),
            ndims_params=[(), ()],  # Both beta and gamma are scalars
        )

        # Sample using SMC
        print("\n🚀 Starting ABC-SMC sampling...")
        print("   (This may take several minutes...)\n")

        trace = pm.sample_smc(
            draws=n_samples,
            chains=1,
            progressbar=True,
        )

    print("\n✅ Sampling complete!")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save trace
    trace_path = os.path.join(output_dir, 'abc_trace.pkl')
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)
    print(f"\n📁 Saved trace to: {trace_path}")

    # Convert to InferenceData for arviz
    idata = az.from_pymc(trace)
    netcdf_path = os.path.join(output_dir, 'abc_trace.nc')
    idata.to_netcdf(netcdf_path)
    print(f"📁 Saved InferenceData to: {netcdf_path}")

    return trace, model


def summarize_results(trace, true_beta=None, true_gamma=None):
    """
    Print summary of calibration results.

    Args:
        trace: PyMC trace object
        true_beta (float): True beta value (if known, for validation)
        true_gamma (float): True gamma value (if known, for validation)
    """
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    # Extract posterior samples
    beta_samples = trace.posterior['beta'].values.flatten()
    gamma_samples = trace.posterior['gamma'].values.flatten()
    r0_samples = beta_samples / gamma_samples

    # Calculate statistics
    print("\nPosterior Statistics:")
    print("-" * 70)

    print(f"\nβ (transmission rate):")
    print(f"  Mean:   {np.mean(beta_samples):.4f}")
    print(f"  Median: {np.median(beta_samples):.4f}")
    print(f"  95% CI: [{np.percentile(beta_samples, 2.5):.4f}, "
          f"{np.percentile(beta_samples, 97.5):.4f}]")
    if true_beta is not None:
        print(f"  True:   {true_beta:.4f} ⭐")

    print(f"\nγ (recovery rate):")
    print(f"  Mean:   {np.mean(gamma_samples):.4f}")
    print(f"  Median: {np.median(gamma_samples):.4f}")
    print(f"  95% CI: [{np.percentile(gamma_samples, 2.5):.4f}, "
          f"{np.percentile(gamma_samples, 97.5):.4f}]")
    if true_gamma is not None:
        print(f"  True:   {true_gamma:.4f} ⭐")

    print(f"\nR₀ (derived):")
    print(f"  Mean:   {np.mean(r0_samples):.4f}")
    print(f"  Median: {np.median(r0_samples):.4f}")
    print(f"  95% CI: [{np.percentile(r0_samples, 2.5):.4f}, "
          f"{np.percentile(r0_samples, 97.5):.4f}]")
    if true_beta is not None and true_gamma is not None:
        true_r0 = true_beta / true_gamma
        print(f"  True:   {true_r0:.4f} ⭐")

    # Calculate coverage (if true values known)
    if true_beta is not None:
        beta_ci = np.percentile(beta_samples, [2.5, 97.5])
        beta_covered = beta_ci[0] <= true_beta <= beta_ci[1]
        print(f"\n✓ True β {'IS' if beta_covered else 'IS NOT'} in 95% CI")

    if true_gamma is not None:
        gamma_ci = np.percentile(gamma_samples, [2.5, 97.5])
        gamma_covered = gamma_ci[0] <= true_gamma <= gamma_ci[1]
        print(f"✓ True γ {'IS' if gamma_covered else 'IS NOT'} in 95% CI")


def main():
    """Main calibration pipeline."""
    print("=" * 70)
    print("SIR MODEL CALIBRATION WITH ABC")
    print("=" * 70)

    # Load observed data
    print("\n📊 Loading observed data...")
    df_obs, obs_stats = load_observed_data('data/observed_epidemic.csv')

    # Try to load true parameters from metadata (if available)
    try:
        # Read CSV to check for metadata comments
        with open('data/observed_epidemic.csv', 'r') as f:
            first_line = f.readline()
            if 'true_beta' in first_line:
                # Parse metadata (this is a hack, better to use separate metadata file)
                true_beta = 0.35  # Default from generate_data.py
                true_gamma = 0.15
            else:
                true_beta = None
                true_gamma = None
    except:
        true_beta = None
        true_gamma = None

    # Run ABC calibration
    print("\n🔬 Starting calibration...")
    trace, model = run_abc_calibration(
        observed_stats=obs_stats,
        num_agents=100,
        num_infected=1,
        n_samples=1000,  # Start with moderate number
        n_steps=3,  # 3 SMC iterations
        epsilon=None  # Adaptive threshold
    )

    # Summarize results
    summarize_results(trace, true_beta=true_beta, true_gamma=true_gamma)

    print("\n" + "=" * 70)
    print("✅ CALIBRATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Visualize results: python calibration/visualize_results.py")
    print("  2. Check posterior plots in results/")
    print("  3. Validate with new simulations")


if __name__ == '__main__':
    main()