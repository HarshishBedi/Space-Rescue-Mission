import numpy as np
from scipy.signal import convolve2d


def initialize_belief_matrix_for_one_alien(ship_layout: list[list[str]],
                                           bot_position: tuple[int, int], k: int) -> np.ndarray:
    ship_dim = len(ship_layout)
    ship_layout_array = np.array(ship_layout)
    open_cells = ship_layout_array != 'C'

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    # Initialize the belief matrix
    belief_matrix = np.zeros((ship_dim, ship_dim))
    outside_square = ~((np.arange(ship_dim) >= top) & (np.arange(ship_dim) <= bottom))[:, None] | ~(
            (np.arange(ship_dim) >= left) & (np.arange(ship_dim) <= right))
    valid_cells = open_cells & outside_square
    n = valid_cells.sum()
    belief_matrix[valid_cells] = 1 / n
    return belief_matrix


def get_observation_matrix_for_one_alien(ship_layout: list[list[str]], bot_position: tuple[int, int],
                                         k: int, alien_sensed: bool) -> np.ndarray:
    ship_dim = len(ship_layout)
    ship_layout_array = np.array(ship_layout)
    open_cells = ship_layout_array != 'C'

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    # Set the observation matrix values based on whether the alien is sensed
    observation_matrix = np.zeros((ship_dim, ship_dim))
    if alien_sensed:
        inside_square = ((np.arange(ship_dim) >= top) & (np.arange(ship_dim) <= bottom))[:, None] & (
                (np.arange(ship_dim) >= left) & (np.arange(ship_dim) <= right))
        valid_cells = open_cells & inside_square
    else:
        outside_square = ~((np.arange(ship_dim) >= top) & (np.arange(ship_dim) <= bottom))[:, None] | ~(
                (np.arange(ship_dim) >= left) & (np.arange(ship_dim) <= right))
        valid_cells = open_cells & outside_square
    observation_matrix[valid_cells] = 1
    total_probability = observation_matrix.sum()
    observation_matrix /= total_probability
    return observation_matrix


def update_belief_matrix_for_one_alien(belief_matrix_for_one_alien: np.ndarray,
                                       ship_layout: list[list[str]],
                                       bot_position: tuple[int, int],
                                       k: int, alien_sensed: bool) -> np.ndarray:
    ship_dim = len(ship_layout)
    ship_layout_array = np.array(ship_layout)
    open_cells = ship_layout_array != 'C'

    observation_matrix = get_observation_matrix_for_one_alien(ship_layout, bot_position, k, alien_sensed)

    # Calculate the transition probabilities matrix based on the number of open neighbors around each cell
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    open_neighbors = convolve2d(open_cells, kernel, mode='same', boundary='fill', fillvalue=0)
    transition_probabilities = np.where(open_neighbors > 0, 1 / open_neighbors, 0)

    # Update the belief matrix
    updated_belief_matrix = np.zeros((ship_dim, ship_dim))
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted_belief_matrix = np.roll(np.roll(belief_matrix_for_one_alien, dx, axis=0), dy, axis=1)
        shifted_transition_probabilities = np.roll(np.roll(transition_probabilities, dx, axis=0), dy, axis=1)
        updated_belief_matrix += shifted_belief_matrix * shifted_transition_probabilities * observation_matrix

    # Normalize the updated belief matrix
    total_probability = np.sum(updated_belief_matrix)
    updated_belief_matrix /= total_probability

    return updated_belief_matrix

