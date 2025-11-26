# tabs/tab_simulation.py
"""Tab 1: Interactive Simulation Interface"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import (GRID_SIZE, STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED,
                    COLOR_SUSCEPTIBLE, COLOR_INFECTED, COLOR_RECOVERED, COLOR_EMPTY)
from helper import get_summary_statistics


def render():
    """Render the simulation tab."""

    if not st.session_state.initialized:
        st.info("Configure parameters and click **Start** to begin simulation")
        return

    # Display metrics
    s, i, r = st.session_state.env.get_state_counts()
    total = s + i + r

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Day", st.session_state.step)

    with col2:
        st.metric("Susceptible", f"{s} ({s / total * 100:.1f}%)" if total > 0 else "0")

    with col3:
        st.metric("Infected", f"{i} ({i / total * 100:.1f}%)" if total > 0 else "0")

    with col4:
        st.metric("Recovered", f"{r} ({r / total * 100:.1f}%)" if total > 0 else "0")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        _plot_spatial_grid()

    with col2:
        _plot_time_series()

    # Summary statistics
    if len(st.session_state.history) > 5:
        with st.expander("📈 Current Statistics"):
            stats = get_summary_statistics(st.session_state.history, st.session_state.gamma)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Infected", f"{stats['peak_infected']}")
                st.metric("Peak Day", f"{stats['peak_time']}")

            with col2:
                st.metric("Attack Rate", f"{stats['attack_rate'] * 100:.1f}%")
                st.metric("R₀ Estimate", f"{stats['r0_estimate']:.2f}")

            with col3:
                st.metric("Duration", f"{stats['duration']} days")
                st.metric("Final Susceptible", f"{stats['final_susceptible']}")

    # Information panel
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### SIR Epidemic Model

        Agent-based implementation of the classic Susceptible-Infected-Recovered 
        compartmental model with spatial dynamics.

        **Model Dynamics:**
        - **S → I**: Susceptible agents become infected through contact with infected neighbors
        - **I → R**: Infected agents recover after infectious period
        - **Movement**: Agents move randomly, creating dynamic contact networks

        **Key Parameters:**
        - **β (Beta)**: Transmission probability per contact (0-1)
        - **γ (Gamma)**: Recovery rate = 1 / infectious_period
        - **R₀**: Basic reproduction number = β/γ (expected infections per infected person)

        **Model Predictions:**
        - **R₀ < 1**: Epidemic dies out
        - **R₀ > 1**: Epidemic spreads through population
        - **Final size**: Depends on R₀ and population mixing

        **Key Metrics:**
        - **Attack Rate**: Proportion ultimately infected
        - **Peak Time**: When infections peak
        - **Duration**: Time until epidemic ends
        """)


def _plot_spatial_grid():
    """Plot spatial grid visualization."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Create grid representation
    grid_visual = np.zeros((GRID_SIZE, GRID_SIZE))

    for agent in st.session_state.agents:
        if agent.state == STATE_SUSCEPTIBLE:
            grid_visual[agent.x, agent.y] = 1
        elif agent.state == STATE_INFECTED:
            grid_visual[agent.x, agent.y] = 2
        elif agent.state == STATE_RECOVERED:
            grid_visual[agent.x, agent.y] = 3

    # Custom colormap
    cmap = ListedColormap([COLOR_EMPTY, COLOR_SUSCEPTIBLE,
                           COLOR_INFECTED, COLOR_RECOVERED])

    # Plot
    ax.imshow(grid_visual.T, cmap=cmap, origin='lower',
              interpolation='nearest', vmin=0, vmax=3)

    ax.set_title("Population Spatial Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(False)

    # Legend
    legend_elements = [
        Patch(facecolor=COLOR_SUSCEPTIBLE, label='Susceptible'),
        Patch(facecolor=COLOR_INFECTED, label='Infected'),
        Patch(facecolor=COLOR_RECOVERED, label='Recovered')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    st.pyplot(fig)
    plt.close()


def _plot_time_series():
    """Plot time series of SIR dynamics."""
    if len(st.session_state.history) == 0:
        st.info("Run simulation to see epidemic curves")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=100)

    # Extract time series
    steps = list(range(len(st.session_state.history)))
    s_counts = [h[0] for h in st.session_state.history]
    i_counts = [h[1] for h in st.session_state.history]
    r_counts = [h[2] for h in st.session_state.history]

    # Absolute counts
    ax1.plot(steps, s_counts, color=COLOR_SUSCEPTIBLE, linewidth=2, label='S')
    ax1.plot(steps, i_counts, color=COLOR_INFECTED, linewidth=2, label='I')
    ax1.plot(steps, r_counts, color=COLOR_RECOVERED, linewidth=2, label='R')

    ax1.set_title("SIR Dynamics (Counts)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Number of Individuals")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Proportions (stacked area)
    total_pop = s_counts[0] + i_counts[0] + r_counts[0]
    s_prop = np.array(s_counts) / total_pop
    i_prop = np.array(i_counts) / total_pop
    r_prop = np.array(r_counts) / total_pop

    ax2.fill_between(steps, 0, s_prop, color=COLOR_SUSCEPTIBLE, alpha=0.7, label='S')
    ax2.fill_between(steps, s_prop, s_prop + i_prop,
                     color=COLOR_INFECTED, alpha=0.7, label='I')
    ax2.fill_between(steps, s_prop + i_prop, s_prop + i_prop + r_prop,
                     color=COLOR_RECOVERED, alpha=0.7, label='R')

    ax2.set_title("SIR Dynamics (Proportions)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Proportion of Population")
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()