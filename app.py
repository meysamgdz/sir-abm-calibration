# app.py
"""
SIR Epidemic Model - Streamlit Application

Interactive visualization of spatial SIR epidemic dynamics demonstrating
how diseases spread through populations via contact networks.
"""

import streamlit as st
import random

from agent.agent import Agent
from environment.environment import Environment
from helper import create_agents, place_agents_random
from config import DEFAULT_BETA, DEFAULT_GAMMA, INITIAL_POPULATION, INITIAL_INFECTED

# Import tab modules
from tabs import tab_simulation, tab_analysis, tab_calibration

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(layout="wide", page_title="SIR Epidemic Model")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.running = False
    st.session_state.step = 0
    st.session_state.history = []
    st.session_state.agents = []
    st.session_state.env = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_simulation(num_agents, num_infected, beta, gamma, infectious_period):
    """Initialize or reset the simulation."""
    st.session_state.env = Environment()
    st.session_state.agents = create_agents(num_agents, num_infected, infectious_period)
    place_agents_random(st.session_state.agents, st.session_state.env)

    st.session_state.beta = beta
    st.session_state.gamma = gamma

    st.session_state.step = 0
    st.session_state.history = []
    st.session_state.initialized = True


def run_step():
    """Execute one simulation step."""
    counts = st.session_state.env.get_state_counts()
    st.session_state.history.append(counts)

    if counts[1] == 0:
        st.session_state.running = False
        return

    random.shuffle(st.session_state.agents)

    for agent in st.session_state.agents:
        agent.move(st.session_state.env)
        agent.try_infect_neighbors(st.session_state.env, st.session_state.beta)
        agent.try_recover(st.session_state.env.current_step, st.session_state.gamma)

    st.session_state.env.increment_step()
    st.session_state.step += 1


# ============================================================================
# UI - TITLE
# ============================================================================

st.title("🦠 SIR Epidemic Model")
st.markdown("Agent-based simulation of infectious disease spread with ABC calibration")

# ============================================================================
# UI - SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Model Parameters")

    st.subheader("Population")
    col1, col2 = st.columns(2)
    num_agents = col1.slider("Total Population", 100, 2000, 500, 50)
    infected_pcg = col2.slider("% Infected", 5, 100, 10, 5) * 0.01
    num_infected = int(infected_pcg * num_agents)

    st.subheader("Disease Parameters")
    col1, col2 = st.columns(2)
    beta = col1.slider("Transmission Rate (β)", 0.0, 1.0, 0.3, 0.05,
                       help="Probability of infection per contact")
    infectious_period = col2.slider("Infectious Period (days)", 1, 30, 10, 1,
                                    help="Mean days infected person remains contagious")
    gamma = 1.0 / infectious_period

    col1, col2 = st.columns(2)
    col1.metric("Recovery Rate (γ)", f"{gamma:.3f}")
    col2.metric("Estimated R₀", f"{beta / gamma:.2f}",
                help="Basic reproduction number")

    # Control buttons
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎬 Start", use_container_width=True):
            if not st.session_state.initialized:
                initialize_simulation(num_agents, num_infected, beta, gamma, infectious_period)
            st.session_state.running = True

    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.running = False

    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            initialize_simulation(num_agents, num_infected, beta, gamma, infectious_period)
            st.session_state.running = False

    steps_per_update = st.slider("Steps per Update", 1, 10, 1, 1)

# ============================================================================
# UI - TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🎮 Simulation", "📊 Analysis", "🎯 Calibration"])

with tab1:
    tab_simulation.render()

with tab2:
    tab_analysis.render()

with tab3:
    tab_calibration.render()

# ============================================================================
# AUTO-RUN LOGIC
# ============================================================================

if st.session_state.running:
    for _ in range(steps_per_update):
        run_step()
    st.rerun()