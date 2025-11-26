# tabs/tab_calibration.py
"""Tab 3: ABC Calibration Interface"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import COLOR_SUSCEPTIBLE, COLOR_INFECTED, COLOR_RECOVERED
from helper import run_sir_simulation

# Import calibration modules
try:
    from calibration.summary_stats import calculate_core_statistics
    from calibration.abc_calibration import sir_simulator

    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False


def render():
    """Render the calibration tab."""
    st.header("🎯 ABC Calibration")

    if not CALIBRATION_AVAILABLE:
        st.error("⚠️ Calibration requires PyMC. Install with: `pip install pymc arviz`")
        return

    st.markdown("""
    Calibrate SIR model parameters (β and γ) using Approximate Bayesian Computation (ABC).

    **How it works:**
    1. Upload or generate observed epidemic data
    2. Define prior distributions for β and γ
    3. Run ABC to find posterior distributions
    4. Visualize results and model fit
    """)

    # Initialize calibration state
    if 'calibration_data' not in st.session_state:
        st.session_state.calibration_data = None
        st.session_state.calibration_results = None

    # Render sections
    _data_input_section()
    _calibration_settings_section()
    _results_section()


def _data_input_section():
    """Data input section."""
    st.subheader("📊 Step 1: Data Input")

    data_source = st.radio(
        "Choose data source:",
        ["Generate Synthetic Data", "Use Current Simulation", "Upload CSV"],
        horizontal=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if data_source == "Generate Synthetic Data":
            _generate_synthetic_data()
        elif data_source == "Use Current Simulation":
            _use_current_simulation()
        else:
            _upload_csv()

    with col2:
        if st.session_state.calibration_data is not None:
            _preview_data()


def _generate_synthetic_data():
    """Generate synthetic epidemic data."""
    st.markdown("**Generate synthetic 'observed' data:**")

    subcol1, subcol2 = st.columns(2)
    true_beta_gen = subcol1.slider("True β", 0.1, 0.8, 0.35, 0.05, key='gen_beta')
    true_gamma_gen = subcol2.slider("True γ", 0.05, 0.5, 0.15, 0.05, key='gen_gamma')
    gen_pop = subcol1.number_input("Population", 100, 2000, 500, 50, key='gen_pop')
    inf_pcg = subcol2.slider("% of infected", 5, 100, 10, 5, key='infected_pcg') * 0.01
    num_infected = int(inf_pcg * gen_pop)

    if st.button("🎲 Generate Data"):
        with st.spinner("Generating synthetic epidemic..."):
            history, _, _ = run_sir_simulation(
                num_agents=gen_pop,
                num_infected=num_infected,
                beta=true_beta_gen,
                gamma=true_gamma_gen,
                infectious_period=1.0 / true_gamma_gen,
                max_steps=200,
                seed=42
            )

            df = pd.DataFrame(history, columns=['S', 'I', 'R'])
            df['day'] = range(len(df))

            st.session_state.calibration_data = df
            st.session_state.true_beta = true_beta_gen
            st.session_state.true_gamma = true_gamma_gen
            # Store with different names to avoid widget key conflict
            st.session_state.calibration_pop = gen_pop
            st.session_state.calibration_infected = num_infected

            st.success(f"✅ Generated data with β={true_beta_gen}, γ={true_gamma_gen}")
            st.rerun()


def _use_current_simulation():
    """Use current simulation data."""
    if st.session_state.get('initialized') and len(st.session_state.get('history', [])) > 10:
        if st.button("📥 Use Current Data"):
            df = pd.DataFrame(st.session_state.history, columns=['S', 'I', 'R'])
            df['day'] = range(len(df))

            # Calculate population from first timestep
            first_counts = st.session_state.history[0]
            total_pop = sum(first_counts)
            initial_infected = first_counts[1]

            st.session_state.calibration_data = df
            st.session_state.true_beta = None
            st.session_state.true_gamma = None
            st.session_state.calibration_pop = total_pop
            st.session_state.calibration_infected = initial_infected

            st.success("✅ Loaded current simulation data")
            st.rerun()
    else:
        st.warning("⚠️ Run simulation first (at least 10 steps)")


def _upload_csv():
    """Upload CSV data."""
    uploaded_file = st.file_uploader("Upload CSV with columns: S, I, R", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in ['S', 'I', 'R']):
            if 'day' not in df.columns:
                df['day'] = range(len(df))

            # Calculate population from data
            total_pop = int(df['S'].iloc[0] + df['I'].iloc[0] + df['R'].iloc[0])
            initial_infected = int(df['I'].iloc[0])

            st.session_state.calibration_data = df
            st.session_state.true_beta = None
            st.session_state.true_gamma = None
            st.session_state.calibration_pop = total_pop
            st.session_state.calibration_infected = initial_infected

            st.success("✅ Loaded uploaded data")
        else:
            st.error("❌ CSV must have columns: S, I, R")


def _preview_data():
    """Preview loaded data."""
    df_preview = st.session_state.calibration_data

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_preview['day'], df_preview['S'], label='S', color=COLOR_SUSCEPTIBLE)
    ax.plot(df_preview['day'], df_preview['I'], label='I', color=COLOR_INFECTED)
    ax.plot(df_preview['day'], df_preview['R'], label='R', color=COLOR_RECOVERED)
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.set_title("Observed Data")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()
    st.divider()


def _calibration_settings_section():
    """Calibration settings section."""
    st.subheader("⚙️ Step 2: Calibration Settings")

    if st.session_state.calibration_data is None:
        st.info("👆 Load data first to configure calibration")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**β Prior:**")
        beta_min = st.number_input("β min", 0.05, 0.5, 0.05, 0.05, key='beta_min')
        beta_max = st.number_input("β max", 0.1, 1.0, 0.8, 0.05, key='beta_max')

    with col2:
        st.markdown("**γ Prior:**")
        gamma_min = st.number_input("γ min", 0.05, 0.3, 0.05, 0.05, key='gamma_min')
        gamma_max = st.number_input("γ max", 0.1, 1.0, 0.5, 0.05, key='gamma_max')

    with col3:
        st.markdown("**ABC Settings:**")
        n_samples = st.number_input("Samples", 100, 5000, 500, 100,
                                    help="More = better but slower", key='n_samples')

    st.divider()

    # Run calibration button
    st.subheader("🚀 Step 3: Run Calibration")

    if st.button("Start ABC Calibration", type="primary", use_container_width=True):
        _run_calibration(beta_min, beta_max, gamma_min, gamma_max, n_samples)


def _run_calibration(beta_min, beta_max, gamma_min, gamma_max, n_samples):
    """Run ABC calibration."""
    # Use the renamed session state variables
    gen_pop = st.session_state.get('calibration_pop', 500)
    num_infected = st.session_state.get('calibration_infected', 50)

    # Calculate observed statistics
    obs_stats = calculate_core_statistics(
        st.session_state.calibration_data,
        num_agents=gen_pop
    )

    status_placeholder = st.empty()

    try:
        status_placeholder.info("🔄 Running Simple ABC (Rejection Sampling)...")

        # Distance function
        def calc_distance(sim_stats):
            obs_array = np.atleast_1d(obs_stats)
            sim_array = np.atleast_1d(sim_stats)
            std = np.std([obs_array, sim_array], axis=0)
            std = np.where(std > 0, std, 1.0)
            normalized_diff = (obs_array - sim_array) / std
            return np.sqrt(np.sum(normalized_diff ** 2))

        # Sample and simulate
        distances = []
        progress_bar = st.progress(0)

        for i in range(n_samples):
            beta_sample = np.random.uniform(beta_min, beta_max)
            gamma_sample = np.random.uniform(gamma_min, gamma_max)

            sim_stats = sir_simulator(
                beta=beta_sample,
                gamma=gamma_sample,
                num_agents=gen_pop,
                num_infected=num_infected,
                max_steps=100,
            )

            dist = calc_distance(sim_stats)
            distances.append((dist, beta_sample, gamma_sample))

            if i % 10 == 0:
                progress_bar.progress((i + 1) / n_samples)

        progress_bar.progress(1.0)

        # Keep best 20%
        distances.sort(key=lambda x: x[0])
        n_accept = max(int(n_samples * 0.2), 50)
        accepted = distances[:n_accept]

        accepted_beta = [x[1] for x in accepted]
        accepted_gamma = [x[2] for x in accepted]

        # Store results
        class SimpleTrace:
            def __init__(self, posterior):
                self.posterior = posterior

        st.session_state.calibration_results = SimpleTrace({
            'beta': np.array(accepted_beta).reshape(1, -1, 1),
            'gamma': np.array(accepted_gamma).reshape(1, -1, 1)
        })

        status_placeholder.success(f"✅ Calibration complete! Accepted {n_accept}/{n_samples} samples")
        st.rerun()

    except Exception as e:
        status_placeholder.error(f"❌ Calibration failed: {str(e)}")
        st.exception(e)


def _results_section():
    """Results section."""
    st.divider()
    st.subheader("📈 Step 4: Results")

    if st.session_state.calibration_results is None:
        st.info("👆 Run calibration to see results")
        return

    trace = st.session_state.calibration_results

    # Extract samples
    beta_samples = trace.posterior['beta'].flatten()
    gamma_samples = trace.posterior['gamma'].flatten()
    r0_samples = beta_samples / gamma_samples

    # Display statistics
    _display_statistics(beta_samples, gamma_samples, r0_samples)

    st.markdown("---")

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        _plot_posteriors(beta_samples, gamma_samples, r0_samples)

    with col2:
        _plot_joint_posterior(beta_samples, gamma_samples)

    st.markdown("---")

    # Model fit
    _plot_model_fit(beta_samples, gamma_samples)


def _display_statistics(beta_samples, gamma_samples, r0_samples):
    """Display posterior statistics."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**β (Transmission Rate)**")
        st.metric("Median", f"{np.median(beta_samples):.4f}")
        ci = np.percentile(beta_samples, [2.5, 97.5])
        st.metric("95% CI", f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        if hasattr(st.session_state, 'true_beta') and st.session_state.true_beta:
            st.metric("True", f"{st.session_state.true_beta:.4f}")

    with col2:
        st.markdown("**γ (Recovery Rate)**")
        st.metric("Median", f"{np.median(gamma_samples):.4f}")
        ci = np.percentile(gamma_samples, [2.5, 97.5])
        st.metric("95% CI", f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        if hasattr(st.session_state, 'true_gamma') and st.session_state.true_gamma:
            st.metric("True", f"{st.session_state.true_gamma:.4f}")

    with col3:
        st.markdown("**R₀ (Basic Reproduction)**")
        st.metric("Median", f"{np.median(r0_samples):.2f}")
        ci = np.percentile(r0_samples, [2.5, 97.5])
        st.metric("95% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}]")
        if hasattr(st.session_state, 'true_beta') and st.session_state.true_beta and st.session_state.true_gamma:
            st.metric("True", f"{st.session_state.true_beta / st.session_state.true_gamma:.2f}")


def _plot_posteriors(beta_samples, gamma_samples, r0_samples):
    """Plot posterior histograms."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 6))

    # Beta
    axes[0].hist(beta_samples, bins=15, density=True, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].axvline(np.median(beta_samples), color='red', linestyle='--', linewidth=2, label='Median')
    if hasattr(st.session_state, 'true_beta') and st.session_state.true_beta:
        axes[0].axvline(st.session_state.true_beta, color='green', linestyle='--', linewidth=2, label='True')
    axes[0].set_xlabel('β')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Posterior: β')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Gamma
    axes[1].hist(gamma_samples, bins=15, density=True, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[1].axvline(np.median(gamma_samples), color='red', linestyle='--', linewidth=2)
    if hasattr(st.session_state, 'true_gamma') and st.session_state.true_gamma:
        axes[1].axvline(st.session_state.true_gamma, color='green', linestyle='--', linewidth=2)
    axes[1].set_xlabel('γ')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Posterior: γ')
    axes[1].grid(alpha=0.3)

    # R0
    axes[2].hist(r0_samples, bins=30, density=True, alpha=0.7, color='#2ecc71', edgecolor='black')
    axes[2].axvline(np.median(r0_samples), color='red', linestyle='--', linewidth=2)
    if hasattr(st.session_state, 'true_beta') and st.session_state.true_beta and st.session_state.true_gamma:
        axes[2].axvline(st.session_state.true_beta / st.session_state.true_gamma, color='green',
                        linestyle='--', linewidth=2)
    axes[2].set_xlabel('R₀')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Posterior: R₀')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _plot_joint_posterior(beta_samples, gamma_samples):
    """Plot joint posterior."""
    fig, ax = plt.subplots(figsize=(5, 5))

    h = ax.hexbin(beta_samples, gamma_samples, gridsize=20, cmap='Blues', mincnt=1)
    ax.scatter(np.median(beta_samples), np.median(gamma_samples),
               s=100, c='red', marker='o', edgecolor='white', linewidth=2, label='Median', zorder=10)
    if hasattr(st.session_state, 'true_beta') and st.session_state.true_beta and st.session_state.true_gamma:
        ax.scatter(st.session_state.true_beta, st.session_state.true_gamma,
                   s=200, c='green', marker='*', edgecolor='white', linewidth=2, label='True', zorder=10)

    ax.set_xlabel('β (Transmission Rate)')
    ax.set_ylabel('γ (Recovery Rate)')
    ax.set_title('Joint Posterior Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(h, ax=ax, label='Density')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _plot_model_fit(beta_samples, gamma_samples):
    """Plot model fit comparison."""
    st.markdown("**Model Fit Comparison:**")

    # Use the renamed session state variables
    gen_pop = st.session_state.get('calibration_pop', 500)
    num_infected = st.session_state.get('calibration_infected', 50)

    with st.spinner("Generating posterior predictive simulations..."):
        n_post_sims = min(50, len(beta_samples))
        indices = np.random.choice(len(beta_samples), size=n_post_sims, replace=False)

        all_i_curves = []

        for idx in indices:
            beta_sim = float(beta_samples[idx])
            gamma_sim = float(gamma_samples[idx])

            if beta_sim <= 0 or gamma_sim <= 0:
                continue

            history, _, _ = run_sir_simulation(
                num_agents=gen_pop,
                num_infected=num_infected,
                beta=beta_sim,
                gamma=gamma_sim,
                infectious_period=1.0 / gamma_sim,
                max_steps=len(st.session_state.calibration_data),
                seed=None
            )

            i_curve = [h[1] for h in history]

            if len(i_curve) < len(st.session_state.calibration_data):
                i_curve.extend([i_curve[-1]] * (len(st.session_state.calibration_data) - len(i_curve)))

            i_curve = i_curve[:len(st.session_state.calibration_data)]

            if max(i_curve) > 0:
                all_i_curves.append(i_curve)

        if len(all_i_curves) == 0:
            st.error("⚠️ No valid simulations generated.")
            st.write(f"Beta range: [{np.min(beta_samples):.4f}, {np.max(beta_samples):.4f}]")
            st.write(f"Gamma range: [{np.min(gamma_samples):.4f}, {np.max(gamma_samples):.4f}]")
            return

        all_i_curves = np.array(all_i_curves)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))

        days = np.arange(len(st.session_state.calibration_data))

        ax.fill_between(days,
                        np.percentile(all_i_curves, 2.5, axis=0),
                        np.percentile(all_i_curves, 97.5, axis=0),
                        alpha=0.3, color=COLOR_INFECTED, label='95% Credible Interval')

        ax.plot(days, np.median(all_i_curves, axis=0),
                color=COLOR_INFECTED, linewidth=2, label='Posterior Median')

        ax.plot(days, st.session_state.calibration_data['I'],
                'ko-', linewidth=2, markersize=3, label='Observed', alpha=0.7)

        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Infected', fontsize=12)
        ax.set_title('Model Fit: Infected Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close()

        st.write(f"Generated {len(all_i_curves)} valid simulations out of {n_post_sims} attempts")