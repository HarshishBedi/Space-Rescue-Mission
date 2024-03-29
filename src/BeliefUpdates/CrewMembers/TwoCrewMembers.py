import math
import time

import numpy as np

from src.Utilities.utility import get_num_of_open_cells_in_ship, get_manhattan_distance


def initialize_belief_matrix_for_two_crew_members(ship_layout: list[list[str]]) -> np.ndarray:
    n = get_num_of_open_cells_in_ship(ship_layout)
    ship_dim = len(ship_layout)
    belief_matrix_for_two_crew_members = np.zeros((ship_dim, ship_dim, ship_dim, ship_dim), float)

    for i1 in range(ship_dim):
        for j1 in range(ship_dim):
            for i2 in range(ship_dim):
                for j2 in range(ship_dim):
                    if (ship_layout[i1][j1] not in ['C', 'B'] and ship_layout[i2][j2] not in ['C', 'B']
                            and (i1, j1) != (i2, j2)):
                        # As two crew members can't be in the same cell
                        belief_matrix_for_two_crew_members[i1][j1][i2][j2] = 1 / ((n - 1) * (n - 2))
    return belief_matrix_for_two_crew_members


def get_observation_matrix_for_two_crew_members(ship_layout, bot_position, alpha, is_beep):
    ship_dim = len(ship_layout)
    i1, j1, i2, j2 = np.ogrid[:ship_dim, :ship_dim, :ship_dim, :ship_dim]
    d1 = np.abs(i1 - bot_position[0]) + np.abs(j1 - bot_position[1])
    d2 = np.abs(i2 - bot_position[0]) + np.abs(j2 - bot_position[1])
    beep_prob_1 = np.exp(-alpha * (d1 - 1))
    beep_prob_2 = np.exp(-alpha * (d2 - 1))

    if is_beep:
        observation_matrix = beep_prob_1 + beep_prob_2 - beep_prob_1 * beep_prob_2
    else:
        observation_matrix = (1 - beep_prob_1) * (1 - beep_prob_2)

    return observation_matrix


def update_belief_matrix_for_two_crew_members(belief_matrix: np.ndarray, ship_layout: list[list[str]],
                                              bot_position: tuple[int, int], alpha: float,
                                              is_beep: bool) -> np.ndarray:
    observation_matrix = get_observation_matrix_for_two_crew_members(ship_layout, bot_position, alpha, is_beep)
    updated_belief_matrix = belief_matrix * observation_matrix
    updated_belief_matrix /= updated_belief_matrix.sum()
    return updated_belief_matrix
