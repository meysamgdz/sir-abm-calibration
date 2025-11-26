"""
SIR Agent-Based Model - Configuration

Global parameters for the SIR epidemic model including grid size,
disease parameters, and agent behavior settings.
"""

# ============================================================================
# Grid Configuration
# ============================================================================

GRID_SIZE = 50  # Grid dimensions (50x50)

# ============================================================================
# Disease Parameters
# ============================================================================

# Default transmission probability (β) - probability of infection on contact
DEFAULT_BETA = 0.3

# Default recovery rate (γ) - probability of recovery per time step
# Or equivalently: mean infectious period = 1/γ
DEFAULT_GAMMA = 0.1  # ~10 days infectious period

# Incubation period (for SEIR extension)
INCUBATION_PERIOD = 5  # Days before symptoms/infectiousness

# ============================================================================
# Agent States
# ============================================================================

STATE_SUSCEPTIBLE = 0
STATE_INFECTED = 1
STATE_RECOVERED = 2
STATE_DEAD = 3  # Optional: for mortality extension

# ============================================================================
# Movement Parameters
# ============================================================================

# Probability agent moves each time step
MOVEMENT_PROBABILITY = 0.8

# Movement distance (1 = adjacent cells, 2 = up to 2 cells away)
MOVEMENT_RANGE = 1

# ============================================================================
# Initial Conditions
# ============================================================================

INITIAL_POPULATION = 100
INITIAL_INFECTED = 1  # Patient zero

# ============================================================================
# Visualization Colors
# ============================================================================

COLOR_SUSCEPTIBLE = '#4A90E2'  # Blue
COLOR_INFECTED = '#E74C3C'     # Red
COLOR_RECOVERED = '#2ECC71'    # Green
COLOR_DEAD = '#34495E'         # Dark gray
COLOR_EMPTY = '#FFFFFF'        # White

# ============================================================================
# Contact/Interaction
# ============================================================================

# Contact radius - how close agents must be to interact
CONTACT_RADIUS = 1  # Moore neighborhood (8 adjacent cells)

# Contact probability - probability of attempting contact if in range
CONTACT_PROBABILITY = 1.0  # Always attempt if in range