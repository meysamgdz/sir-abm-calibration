"""
Generate Synthetic Observed Data

Creates "ground truth" epidemic data by running the SIR model with known parameters.
This synthetic data will be used as the calibration target for ABC.

In practice, replace this with real epidemic data (e.g., COVID-19, flu outbreaks).
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import run_sir_simulation


def generate_synthetic_data(true_beta=0.35, true_gamma=0.15,
                            num_agents=100, num_infected=1,
                            max_steps=200, seed=42,
                            save_path='data/observed_epidemic.csv'):
    """
    Generate synthetic epidemic data with known parameters.

    Args:
        true_beta (float): True transmission rate (ground truth)
        true_gamma (float): True recovery rate (ground truth)
        num_agents (int): Population size
        num_infected (int): Initial infected count
        max_steps (int): Maximum simulation steps
        seed (int): Random seed for reproducibility
        save_path (str): Path to save CSV file

    Returns:
        pd.DataFrame: Time series of S, I, R counts
    """
    print(f"Generating synthetic data with:")
    print(f"  β (beta) = {true_beta}")
    print(f"  γ (gamma) = {true_gamma}")
    print(f"  R₀ = {true_beta / true_gamma:.2f}")
    print(f"  Population = {num_agents}")
    print(f"  Initial infected = {num_infected}")

    # Run simulation
    infectious_period = 1.0 / true_gamma
    history, env, agents = run_sir_simulation(
        num_agents=num_agents,
        num_infected=num_infected,
        beta=true_beta,
        gamma=true_gamma,
        infectious_period=infectious_period,
        max_steps=max_steps,
        seed=seed
    )

    # Convert to DataFrame
    df = pd.DataFrame(history, columns=['S', 'I', 'R'])
    df['day'] = range(len(df))

    # Add metadata as attributes
    df.attrs['true_beta'] = true_beta
    df.attrs['true_gamma'] = true_gamma
    df.attrs['r0'] = true_beta / true_gamma
    df.attrs['population'] = num_agents
    df.attrs['seed'] = seed

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    df.to_csv(save_path, index=False)

    print(f"\n✅ Saved to: {save_path}")
    print(f"   Duration: {len(df)} days")
    print(f"   Peak infected: {df['I'].max()}")
    print(f"   Attack rate: {df['R'].iloc[-1] / num_agents * 100:.1f}%")

    return df


def generate_multiple_scenarios(output_dir='data/scenarios'):
    """
    Generate multiple synthetic datasets with different parameters.
    Useful for testing calibration robustness.

    Args:
        output_dir (str): Directory to save scenario files
    """
    scenarios = [
        {'name': 'mild', 'beta': 0.2, 'gamma': 0.15},  # R₀ ≈ 1.3
        {'name': 'moderate', 'beta': 0.35, 'gamma': 0.15},  # R₀ ≈ 2.3
        {'name': 'severe', 'beta': 0.5, 'gamma': 0.1},  # R₀ = 5.0
    ]

    os.makedirs(output_dir, exist_ok=True)

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario['name'].upper()}")
        print(f"{'=' * 60}")

        save_path = f"{output_dir}/{scenario['name']}_epidemic.csv"
        df = generate_synthetic_data(
            true_beta=scenario['beta'],
            true_gamma=scenario['gamma'],
            save_path=save_path,
            seed=42
        )


if __name__ == '__main__':
    print("=" * 60)
    print("SYNTHETIC EPIDEMIC DATA GENERATOR")
    print("=" * 60)

    # Generate main observed data
    print("\nGenerating main 'observed' dataset...")
    df = generate_synthetic_data(
        true_beta=0.35,  # True transmission rate
        true_gamma=0.15,  # True recovery rate (infectious period ≈ 6.7 days)
        num_agents=100,
        num_infected=1,
        seed=42
    )

    # Optionally generate multiple scenarios
    print("\n\nGenerating additional scenarios...")
    generate_multiple_scenarios()

    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)