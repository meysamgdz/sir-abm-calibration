"""
SIR Model Helper Functions

Utility functions for agent creation, initialization, and epidemic analysis.
"""

import random
import numpy as np
from agent.agent import Agent
from config import STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED


def create_agents(num_agents, num_infected=1, infectious_period=10):
    """
    Create a population of agents with initial infections.

    Args:
        num_agents (int): Total number of agents to create
        num_infected (int): Number of initially infected agents
        infectious_period (float): Mean infectious period (1/gamma)

    Returns:
        list: List of Agent objects
    """
    agents = []

    # Create susceptible agents
    for _ in range(num_agents - num_infected):
        agent = Agent(0, 0, state=STATE_SUSCEPTIBLE, infectious_period=infectious_period)
        agents.append(agent)

    # Create infected agents (patient zeros)
    for _ in range(num_infected):
        agent = Agent(0, 0, state=STATE_INFECTED, infectious_period=infectious_period)
        agent.infection_time = 0  # Infected at start
        agents.append(agent)

    # Shuffle to randomize
    random.shuffle(agents)

    return agents


def place_agents_random(agents, env):
    """
    Place all agents randomly on the grid.

    Args:
        agents (list): List of Agent objects
        env: Environment object

    Returns:
        int: Number of agents successfully placed
    """
    placed = 0
    for agent in agents:
        if env.place_agent_random(agent):
            placed += 1
    return placed


def calculate_sir_curves(history):
    """
    Extract S, I, R time series from history.

    Args:
        history (list): List of (S, I, R) tuples over time

    Returns:
        tuple: Three numpy arrays (s_curve, i_curve, r_curve)
    """
    if len(history) == 0:
        return np.array([]), np.array([]), np.array([])

    s_curve = np.array([counts[0] for counts in history])
    i_curve = np.array([counts[1] for counts in history])
    r_curve = np.array([counts[2] for counts in history])

    return s_curve, i_curve, r_curve


def calculate_attack_rate(history):
    """
    Calculate final attack rate (proportion of population infected).

    Args:
        history (list): List of (S, I, R) tuples over time

    Returns:
        float: Attack rate (0.0 to 1.0)
    """
    if len(history) == 0:
        return 0.0

    initial_pop = sum(history[0])
    final_recovered = history[-1][2]  # R count at end

    if initial_pop == 0:
        return 0.0

    return final_recovered / initial_pop


def calculate_peak_info(history):
    """
    Find peak infection time and magnitude.

    Args:
        history (list): List of (S, I, R) tuples over time

    Returns:
        tuple: (peak_time, peak_infected, peak_prevalence)
    """
    if len(history) == 0:
        return 0, 0, 0.0

    infected_counts = [counts[1] for counts in history]
    peak_infected = max(infected_counts)
    peak_time = infected_counts.index(peak_infected)

    total_pop = sum(history[peak_time])
    peak_prevalence = peak_infected / total_pop if total_pop > 0 else 0.0

    return peak_time, peak_infected, peak_prevalence


def calculate_r0_estimate(history, gamma):
    """
    Estimate basic reproduction number R0 from early exponential growth.

    Uses approximation: R0 ≈ 1 + (growth_rate / gamma)

    Args:
        history (list): List of (S, I, R) tuples over time
        gamma (float): Recovery rate

    Returns:
        float: Estimated R0
    """
    if len(history) < 10:
        return 0.0

    # Get early infected counts (first 10 steps)
    early_infected = [counts[1] for counts in history[:10]]

    # Calculate exponential growth rate
    log_infected = np.log(np.array(early_infected) + 1)  # +1 to avoid log(0)

    # Linear regression on log scale
    x = np.arange(len(log_infected))
    if len(x) > 1:
        coeffs = np.polyfit(x, log_infected, 1)
        growth_rate = coeffs[0]

        # R0 approximation
        r0 = 1 + (growth_rate / gamma)
        return max(0, r0)  # R0 can't be negative

    return 0.0


def calculate_epidemic_duration(history, threshold=1):
    """
    Calculate duration of epidemic (time until infections < threshold).

    Args:
        history (list): List of (S, I, R) tuples over time
        threshold (int): Minimum infected count to consider epidemic ongoing

    Returns:
        int: Number of time steps until epidemic ends
    """
    for t, counts in enumerate(history):
        infected = counts[1]
        if infected < threshold:
            return t

    return len(history)


def get_summary_statistics(history, gamma):
    """
    Calculate comprehensive epidemic summary statistics.

    Args:
        history (list): List of (S, I, R) tuples over time
        gamma (float): Recovery rate

    Returns:
        dict: Dictionary of summary statistics
    """
    peak_time, peak_infected, peak_prevalence = calculate_peak_info(history)

    return {
        'attack_rate': calculate_attack_rate(history),
        'peak_time': peak_time,
        'peak_infected': peak_infected,
        'peak_prevalence': peak_prevalence,
        'r0_estimate': calculate_r0_estimate(history, gamma),
        'duration': calculate_epidemic_duration(history),
        'final_susceptible': history[-1][0] if history else 0,
        'final_recovered': history[-1][2] if history else 0
    }


def run_sir_simulation(num_agents, num_infected, beta, gamma,
                       infectious_period, max_steps=300, seed=None):
    """
    Run a complete SIR simulation and return time series.

    Args:
        num_agents (int): Total population size
        num_infected (int): Initial infected count
        beta (float): Transmission probability
        gamma (float): Recovery rate
        infectious_period (float): Mean infectious period
        max_steps (int): Maximum simulation steps
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (history, env, agents) where history is list of (S,I,R) counts
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    from environment.environment import Environment

    # Initialize
    env = Environment()
    agents = create_agents(num_agents, num_infected, infectious_period)
    place_agents_random(agents, env)

    # Track history
    history = []

    # Run simulation
    for step in range(max_steps):
        # Record state
        counts = env.get_state_counts()
        history.append(counts)

        # Stop if no more infected
        if counts[1] == 0:
            break

        # Shuffle agents for random order
        random.shuffle(agents)

        # Agent actions
        for agent in agents:
            # Move
            agent.move(env)

            # Attempt transmission
            agent.try_infect_neighbors(env, beta)

            # Attempt recovery
            agent.try_recover(env.current_step, gamma)

        # Increment time
        env.increment_step()

    return history, env, agents