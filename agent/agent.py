"""
SIR Agent Class

Represents an individual agent in the epidemic model with one of three states:
Susceptible, Infected, or Recovered. Agents can move, interact, and transition
between states according to disease dynamics.
"""

import random
from config import (STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED,
                    MOVEMENT_PROBABILITY, MOVEMENT_RANGE, CONTACT_RADIUS)


class Agent:
    """
    Individual agent in SIR epidemic model.

    Attributes:
        x (int): X-coordinate position on grid
        y (int): Y-coordinate position on grid
        state (int): Current disease state (S=0, I=1, R=2)
        infection_time (int): Time step when agent became infected
        infectious_period (float): How long agent remains infectious (1/gamma)
    """

    def __init__(self, x, y, state=STATE_SUSCEPTIBLE, infectious_period=10):
        """
        Initialize agent.

        Args:
            x (int): Initial x position
            y (int): Initial y position
            state (int): Initial disease state (default: susceptible)
            infectious_period (float): Mean days of infectiousness (1/gamma)
        """
        self.x = x
        self.y = y
        self.state = state
        self.infection_time = None
        self.infectious_period = infectious_period

    def is_susceptible(self):
        """Check if agent is susceptible."""
        return self.state == STATE_SUSCEPTIBLE

    def is_infected(self):
        """Check if agent is infected."""
        return self.state == STATE_INFECTED

    def is_recovered(self):
        """Check if agent is recovered."""
        return self.state == STATE_RECOVERED

    def infect(self, current_step):
        """
        Transition agent from susceptible to infected.

        Args:
            current_step (int): Current simulation time step
        """
        if self.is_susceptible():
            self.state = STATE_INFECTED
            self.infection_time = current_step

    def try_recover(self, current_step, gamma):
        """
        Attempt recovery based on time infected and recovery rate.

        Args:
            current_step (int): Current simulation time step
            gamma (float): Recovery rate (probability per time step)

        Returns:
            bool: True if agent recovered this step
        """
        if not self.is_infected():
            return False

        # Calculate time infected
        time_infected = current_step - self.infection_time

        # Recovery probability increases with time infected
        # Using exponential recovery: P(recover) = gamma
        if random.random() < gamma:
            self.state = STATE_RECOVERED
            return True

        return False

    def get_neighbors(self, env, radius=CONTACT_RADIUS):
        """
        Get all agents within contact radius.

        Args:
            env: Environment object
            radius (int): Contact radius (Moore neighborhood distance)

        Returns:
            list: List of Agent objects within radius
        """
        neighbors = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                # Calculate neighbor position with wrapping
                nx = (self.x + dx) % env.grid_size
                ny = (self.y + dy) % env.grid_size

                # Get agent at this position
                agent = env.get_agent_at(nx, ny)
                if agent is not None:
                    neighbors.append(agent)

        return neighbors

    def try_infect_neighbors(self, env, beta):
        """
        Attempt to infect susceptible neighbors.

        Args:
            env: Environment object
            beta (float): Transmission probability per contact

        Returns:
            int: Number of new infections caused
        """
        if not self.is_infected():
            return 0

        neighbors = self.get_neighbors(env)
        infections = 0

        for neighbor in neighbors:
            if neighbor.is_susceptible():
                # Transmission occurs with probability beta
                if random.random() < beta:
                    neighbor.infect(env.current_step)
                    infections += 1

        return infections

    def move(self, env):
        """
        Move agent to a random nearby location.

        Args:
            env: Environment object

        Returns:
            bool: True if move was successful
        """
        # Decide whether to move
        if random.random() > MOVEMENT_PROBABILITY:
            return False

        # Generate random move within range
        dx = random.randint(-MOVEMENT_RANGE, MOVEMENT_RANGE)
        dy = random.randint(-MOVEMENT_RANGE, MOVEMENT_RANGE)

        # Calculate new position with wrapping
        new_x = (self.x + dx) % env.grid_size
        new_y = (self.y + dy) % env.grid_size

        # Check if target cell is empty
        if env.is_empty(new_x, new_y):
            # Update environment
            env.move_agent(self, new_x, new_y)
            return True

        return False

    def __repr__(self):
        """String representation of agent."""
        state_name = {STATE_SUSCEPTIBLE: 'S',
                      STATE_INFECTED: 'I',
                      STATE_RECOVERED: 'R'}
        return f"Agent({self.x}, {self.y}, {state_name[self.state]})"