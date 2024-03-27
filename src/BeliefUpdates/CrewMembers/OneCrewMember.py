import numpy as np


def initialize_belief_matrix_for_one_crew_member(ship_layout: list[list[str]]) -> np.ndarray:
    open_cells = (np.array(ship_layout) != 'C') & (np.array(ship_layout) != 'B')
    n = open_cells.sum()
    belief_matrix = np.where(open_cells, 1 / n, 0)
    return belief_matrix


def get_observation_matrix_for_one_crew_member(ship_layout: list[list[str]], bot_position: tuple[int, int],
                                               alpha: float, is_beep: bool) -> np.ndarray:
    ship_dim = len(ship_layout)
    grid = np.indices((ship_dim, ship_dim)).transpose(1, 2, 0)
    distances = np.abs(grid - bot_position).sum(axis=2)
    if is_beep:
        observation_matrix = np.exp(-alpha * (distances - 1))
    else:
        observation_matrix = 1 - np.exp(-alpha * (distances - 1))
    observation_matrix[ship_layout == 'C'] = 0
    return observation_matrix


def update_belief_matrix_for_one_crew_member(belief_matrix: np.ndarray, ship_layout: list[list[str]],
                                             bot_position: tuple[int, int], alpha: float, is_beep: bool) -> np.ndarray:
    observation_matrix = get_observation_matrix_for_one_crew_member(ship_layout, bot_position, alpha, is_beep)
    updated_belief_matrix = belief_matrix * observation_matrix
    total_probability = updated_belief_matrix.sum()
    updated_belief_matrix /= total_probability
    return updated_belief_matrix
