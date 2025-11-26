"""
Visualize ABC Calibration Results

Creates publication-quality plots of:
1. Posterior distributions for beta and gamma
2. Joint posterior (beta vs gamma)
3. Model fit comparison (observed vs calibrated simulations)
4. Posterior predictive checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import run_sir_simulation

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(trace_path='results/abc_trace.pkl',
                 data_path='data/observed_epidemic.csv'):
    """
    Load calibration results and observed data.

    Args:
        trace_path (str): Path to saved trace
        data_path (str): Path to observed data

    Returns:
        tuple: (trace, observed_df, beta_samples, gamma_samples)
    """
    # Load trace
    with open(trace_path, 'rb') as f:
        trace = pickle.load(f)

    # Extract samples
    beta_samples = trace.posterior['beta'].values.flatten()
    gamma_samples = trace.posterior['gamma'].values.flatten()

    # Load observed data
    df_obs = pd.read_csv(data_path)

    print(f"Loaded {len(beta_samples)} posterior samples")

    return trace, df_obs, beta_samples, gamma_samples


def plot_posteriors(beta_samples, gamma_samples,
                    true_beta=None, true_gamma=None,
                    save_path='results/posteriors.png'):
    """
    Plot marginal posterior distributions.

    Args:
        beta_samples (array): Posterior samples for beta
        gamma_samples (array): Posterior samples for gamma
        true_beta (float): True beta (if known)
        true_gamma (float): True gamma (if known)
        save_path (str): Where to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # R0 = beta / gamma
    r0_samples = beta_samples / gamma_samples

    # Plot beta
    axes[0].hist(beta_samples, bins=40, density=True,
                 alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].axvline(np.median(beta_samples), color='red',
                    linestyle='--', linewidth=2, label='Median')
    if true_beta is not None:
        axes[0].axvline(true_beta, color='green',
                        linestyle='--', linewidth=2, label='True')
    axes[0].set_xlabel('β (Transmission Rate)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Posterior: β', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot gamma
    axes[1].hist(gamma_samples, bins=40, density=True,
                 alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[1].axvline(np.median(gamma_samples), color='red',
                    linestyle='--', linewidth=2, label='Median')
    if true_gamma is not None:
        axes[1].axvline(true_gamma, color='green',
                        linestyle='--', linewidth=2, label='True')
    axes[1].set_xlabel('γ (Recovery Rate)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Posterior: γ', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Plot R0
    axes[2].hist(r0_samples, bins=40, density=True,
                 alpha=0.7, color='#2ecc71', edgecolor='black')
    axes[2].axvline(np.median(r0_samples), color='red',
                    linestyle='--', linewidth=2, label='Median')
    if true_beta is not None and true_gamma is not None:
        true_r0 = true_beta / true_gamma
        axes[2].axvline(true_r0, color='green',
                        linestyle='--', linewidth=2, label='True')
    axes[2].set_xlabel('R₀ (Basic Reproduction Number)', fontsize=12)
    axes[2].set_ylabel('Density', fontsize=12)
    axes[2].set_title('Posterior: R₀ = β/γ', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved posteriors to: {save_path}")
    plt.close()


def plot_joint_posterior(beta_samples, gamma_samples,
                         true_beta=None, true_gamma=None,
                         save_path='results/joint_posterior.png'):
    """
    Plot joint posterior distribution (beta vs gamma).

    Args:
        beta_samples (array): Posterior samples for beta
        gamma_samples (array): Posterior samples for gamma
        true_beta (float): True beta (if known)
        true_gamma (float): True gamma (if known)
        save_path (str): Where to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 2D histogram / hexbin
    h = ax.hexbin(beta_samples, gamma_samples, gridsize=30,
                  cmap='Blues', mincnt=1, alpha=0.8)

    # Add true values
    if true_beta is not None and true_gamma is not None:
        ax.scatter(true_beta, true_gamma, s=200, c='red',
                   marker='*', edgecolor='white', linewidth=2,
                   label='True Values', zorder=10)

    # Add median
    ax.scatter(np.median(beta_samples), np.median(gamma_samples),
               s=100, c='darkblue', marker='o', edgecolor='white',
               linewidth=2, label='Posterior Median', zorder=10)

    ax.set_xlabel('β (Transmission Rate)', fontsize=12)
    ax.set_ylabel('γ (Recovery Rate)', fontsize=12)
    ax.set_title('Joint Posterior Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.colorbar(h, ax=ax, label='Density')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved joint posterior to: {save_path}")
    plt.close()


def plot_model_fit(df_obs, beta_samples, gamma_samples,
                   n_samples=50, save_path='results/model_fit.png'):
    """
    Plot observed data vs posterior predictive simulations.

    Args:
        df_obs (DataFrame): Observed epidemic data
        beta_samples (array): Posterior beta samples
        gamma_samples (array): Posterior gamma samples
        n_samples (int): Number of posterior samples to simulate
        save_path (str): Where to save figure
    """
    print(f"\n🔄 Running {n_samples} posterior predictive simulations...")

    # Randomly select posterior samples
    indices = np.random.choice(len(beta_samples), size=n_samples, replace=False)

    # Store all simulated curves
    all_s_curves = []
    all_i_curves = []
    all_r_curves = []

    for idx in indices:
        beta = beta_samples[idx]
        gamma = gamma_samples[idx]
        infectious_period = 1.0 / gamma

        # Run simulation
        history, _, _ = run_sir_simulation(
            num_agents=100,
            num_infected=1,
            beta=beta,
            gamma=gamma,
            infectious_period=infectious_period,
            max_steps=len(df_obs),
            seed=None
        )

        # Extract curves
        s_curve = [h[0] for h in history]
        i_curve = [h[1] for h in history]
        r_curve = [h[2] for h in history]

        # Pad if shorter than observed
        if len(s_curve) < len(df_obs):
            pad_length = len(df_obs) - len(s_curve)
            s_curve.extend([s_curve[-1]] * pad_length)
            i_curve.extend([0] * pad_length)
            r_curve.extend([r_curve[-1]] * pad_length)

        all_s_curves.append(s_curve[:len(df_obs)])
        all_i_curves.append(i_curve[:len(df_obs)])
        all_r_curves.append(r_curve[:len(df_obs)])

    # Convert to arrays
    all_s_curves = np.array(all_s_curves)
    all_i_curves = np.array(all_i_curves)
    all_r_curves = np.array(all_r_curves)

    # Calculate percentiles
    days = np.arange(len(df_obs))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Susceptible
    axes[0].fill_between(days,
                         np.percentile(all_s_curves, 2.5, axis=0),
                         np.percentile(all_s_curves, 97.5, axis=0),
                         alpha=0.3, color='#3498db', label='95% Credible Interval')
    axes[0].plot(days, np.median(all_s_curves, axis=0),
                 color='#3498db', linewidth=2, label='Posterior Median')
    axes[0].plot(days, df_obs['S'], 'ko-', linewidth=2,
                 markersize=3, label='Observed', alpha=0.7)
    axes[0].set_ylabel('Susceptible', fontsize=12)
    axes[0].set_title('Model Fit: Susceptible', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Infected
    axes[1].fill_between(days,
                         np.percentile(all_i_curves, 2.5, axis=0),
                         np.percentile(all_i_curves, 97.5, axis=0),
                         alpha=0.3, color='#e74c3c', label='95% Credible Interval')
    axes[1].plot(days, np.median(all_i_curves, axis=0),
                 color='#e74c3c', linewidth=2, label='Posterior Median')
    axes[1].plot(days, df_obs['I'], 'ko-', linewidth=2,
                 markersize=3, label='Observed', alpha=0.7)
    axes[1].set_ylabel('Infected', fontsize=12)
    axes[1].set_title('Model Fit: Infected', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Recovered
    axes[2].fill_between(days,
                         np.percentile(all_r_curves, 2.5, axis=0),
                         np.percentile(all_r_curves, 97.5, axis=0),
                         alpha=0.3, color='#2ecc71', label='95% Credible Interval')
    axes[2].plot(days, np.median(all_r_curves, axis=0),
                 color='#2ecc71', linewidth=2, label='Posterior Median')
    axes[2].plot(days, df_obs['R'], 'ko-', linewidth=2,
                 markersize=3, label='Observed', alpha=0.7)
    axes[2].set_xlabel('Day', fontsize=12)
    axes[2].set_ylabel('Recovered', fontsize=12)
    axes[2].set_title('Model Fit: Recovered', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved model fit to: {save_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("VISUALIZING CALIBRATION RESULTS")
    print("=" * 70)

    # Load results
    print("\n📂 Loading results...")
    trace, df_obs, beta_samples, gamma_samples = load_results()

    # True values (if known from synthetic data)
    true_beta = 0.35
    true_gamma = 0.15

    # Generate plots
    print("\n📊 Generating plots...")

    plot_posteriors(beta_samples, gamma_samples,
                    true_beta=true_beta, true_gamma=true_gamma)

    plot_joint_posterior(beta_samples, gamma_samples,
                         true_beta=true_beta, true_gamma=true_gamma)

    plot_model_fit(df_obs, beta_samples, gamma_samples, n_samples=50)

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - results/posteriors.png")
    print("  - results/joint_posterior.png")
    print("  - results/model_fit.png")


if __name__ == '__main__':
    main()