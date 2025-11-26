# tabs/tab_analysis.py
"""Tab 2: Epidemic Analysis and Statistics"""

import streamlit as st
import matplotlib.pyplot as plt

from helper import get_summary_statistics


def render():
    """Render the analysis tab."""
    st.header("📊 Epidemic Analysis")

    if not st.session_state.initialized or len(st.session_state.history) < 10:
        st.info("⚠️ Run a simulation first to see analysis")
        return

    st.subheader("📈 Full Epidemic Statistics")

    stats = get_summary_statistics(st.session_state.history, st.session_state.gamma)

    # Display comprehensive stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Attack Rate", f"{stats['attack_rate'] * 100:.1f}%")
        st.metric("Peak Prevalence", f"{stats['peak_prevalence'] * 100:.1f}%")

    with col2:
        st.metric("Peak Infected", stats['peak_infected'])
        st.metric("Peak Day", stats['peak_time'])

    with col3:
        st.metric("R₀ Estimate", f"{stats['r0_estimate']:.2f}")
        st.metric("Duration", f"{stats['duration']} days")

    with col4:
        st.metric("Final Susceptible", stats['final_susceptible'])
        st.metric("Final Recovered", stats['final_recovered'])

    # Phase diagram
    st.subheader("📉 Phase Diagram")
    _plot_phase_diagram()


def _plot_phase_diagram():
    """Plot S-I phase plane."""
    fig, ax = plt.subplots(figsize=(10, 6))

    s_counts = [h[0] for h in st.session_state.history]
    i_counts = [h[1] for h in st.session_state.history]

    # S-I phase plane
    ax.plot(s_counts, i_counts, linewidth=2, color='#E74C3C')
    ax.scatter(s_counts[0], i_counts[0], s=100, c='green',
               marker='o', label='Start', zorder=5)
    ax.scatter(s_counts[-1], i_counts[-1], s=100, c='red',
               marker='X', label='End', zorder=5)

    ax.set_xlabel("Susceptible", fontsize=12)
    ax.set_ylabel("Infected", fontsize=12)
    ax.set_title("S-I Phase Plane", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close()