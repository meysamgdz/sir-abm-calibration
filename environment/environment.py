"""
SIR Environment Class

Manages the spatial grid where agents live and interact. Tracks agent positions
and provides methods for movement and neighbor detection.
"""

import numpy as np
from config import GRID_SIZE


class Environment:
    """
    Spatial environment for SIR epidemic model.

    Attributes:
        grid_size (int): Size of square grid
        grid (np.ndarray): 2D array storing agent references
        current_step (int): Current simulation time step
    """

    def __init__(self, grid_size=GRID_SIZE):
        """
        Initialize environment.

        Args:
            grid_size (int): Size of square grid (default from config)
        """
        self.grid_size = grid_size
        self.grid = np.empty((grid_size, grid_size), dtype=object)
        self.current_step = 0

    def place_agent(self, agent, x, y):
        """
        Place an agent at specified coordinates.

        Args:
            agent: Agent object to place
            x (int): X-coordinate
            y (int): Y-coordinate

        Returns:
            bool: True if placement successful, False if occupied
        """
        if not self.is_empty(x, y):
            return False

        self.grid[x, y] = agent
        agent.x = x
        agent.y = y
        return True

    def place_agent_random(self, agent):
        """
        Place agent at random empty location.

        Args:
            agent: Agent object to place

        Returns:
            bool: True if placement successful
        """
        empty_cells = self.get_empty_cells()
        if len(empty_cells) == 0:
            return False

        x, y = empty_cells[np.random.randint(len(empty_cells))]
        return self.place_agent(agent, x, y)

    def move_agent(self, agent, new_x, new_y):
        """
        Move agent to new position.

        Args:
            agent: Agent object to move
            new_x (int): New x-coordinate
            new_y (int): New y-coordinate

        Returns:
            bool: True if move successful
        """
        if not self.is_empty(new_x, new_y):
            return False

        # Clear old position
        self.grid[agent.x, agent.y] = None

        # Set new position
        self.grid[new_x, new_y] = agent
        agent.x = new_x
        agent.y = new_y

        return True

    def remove_agent(self, agent):
        """
        Remove agent from environment.

        Args:
            agent: Agent object to remove
        """
        self.grid[agent.x, agent.y] = None

    def is_empty(self, x, y):
        """
        Check if cell is empty.

        Args:
            x (int): X-coordinate
            y (int): Y-coordinate

        Returns:
            bool: True if cell is empty
        """
        return self.grid[x, y] is None

    def get_agent_at(self, x, y):
        """
        Get agent at specified position.

        Args:
            x (int): X-coordinate
            y (int): Y-coordinate

        Returns:
            Agent or None: Agent at position, or None if empty
        """
        return self.grid[x, y]

    def get_empty_cells(self):
        """
        Get list of all empty cell coordinates.

        Returns:
            list: List of (x, y) tuples for empty cells
        """
        empty = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.is_empty(x, y):
                    empty.append((x, y))
        return empty

    def get_all_agents(self):
        """
        Get list of all agents in environment.

        Returns:
            list: List of all Agent objects
        """
        agents = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                agent = self.grid[x, y]
                if agent is not None:
                    agents.append(agent)
        return agents

    def get_occupancy_rate(self):
        """
        Calculate fraction of cells occupied.

        Returns:
            float: Occupancy rate (0.0 to 1.0)
        """
        total_cells = self.grid_size * self.grid_size
        occupied = total_cells - len(self.get_empty_cells())
        return occupied / total_cells

    def get_state_counts(self):
        """
        Count agents in each disease state.

        Returns:
            tuple: (susceptible, infected, recovered) counts
        """
        from config import STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED

        susceptible = 0
        infected = 0
        recovered = 0

        for agent in self.get_all_agents():
            if agent.state == STATE_SUSCEPTIBLE:
                susceptible += 1
            elif agent.state == STATE_INFECTED:
                infected += 1
            elif agent.state == STATE_RECOVERED:
                recovered += 1

        return susceptible, infected, recovered

    def increment_step(self):
        """Increment the current time step."""
        self.current_step += 1

    def __repr__(self):
        """String representation of environment."""
        s, i, r = self.get_state_counts()
        return f"Environment({self.grid_size}x{self.grid_size}, S={s}, I={i}, R={r})"