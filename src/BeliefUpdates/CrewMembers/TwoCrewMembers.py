import math
import time

import numpy as np

from src.Utilities.utility import get_num_of_open_cells_in_ship, get_manhattan_distance


def initialize_belief_matrix_for_two_crew_members(ship_layout: list[list[str]]) -> np.ndarray:
    n = get_num_of_open_cells_in_ship(ship_layout)
    ship_dim = len(ship_layout)
    belief_matrix_for_two_crew_members = np.zeros((ship_dim,ship_dim,ship_dim,ship_dim),float)
        # [[[[
        # 0 for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)]

    for i1 in range(ship_dim):
        for j1 in range(ship_dim):
            for i2 in range(ship_dim):
                for j2 in range(ship_dim):
                    if (ship_layout[i1][j1] not in ['C', 'B'] and ship_layout[i2][j2] not in ['C', 'B']
                            and (i1, j1) != (i2, j2)):
                        # As two crew members can't be in the same cell
                        belief_matrix_for_two_crew_members[i1][j1][i2][j2] = 1 / ((n - 1) * (n - 2))
    return belief_matrix_for_two_crew_members


def get_observation_matrix_for_two_crew_members(ship_layout: list[list[str]], bot_position: tuple[int, int],
                                                alpha: float, is_beep: bool) -> np.ndarray:
    start_time = time.time()
    ship_dim = len(ship_layout)
    observation_matrix_for_two_crew_members: list[list[list[list[float]]]] = [[[[
        0 for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)]
    for i1 in range(ship_dim):
        for j1 in range(ship_dim):
            for i2 in range(ship_dim):
                for j2 in range(ship_dim):
                    beep_probability_for_first_cell = math.exp(
                        -alpha * (get_manhattan_distance(bot_position, (i1, j1)) - 1))
                    beep_probability_for_second_cell = math.exp(
                        -alpha * (get_manhattan_distance(bot_position, (i2, j2)) - 1))
                    sum_of_beep_probabilities = beep_probability_for_first_cell + beep_probability_for_second_cell
                    product_of_beep_probabilities = beep_probability_for_first_cell * beep_probability_for_second_cell
                    probability_of_no_beep = (1 - beep_probability_for_first_cell) * (
                            1 - beep_probability_for_second_cell)
                    if is_beep:
                        observation_matrix_for_two_crew_members[i1][j1][i2][j2] = (sum_of_beep_probabilities -
                                                                                   product_of_beep_probabilities)
                    else:
                        observation_matrix_for_two_crew_members[i1][j1][i2][j2] = probability_of_no_beep
    print(f'Time taken for generating observation matrix:{time.time()-start_time}')
    return np.array(observation_matrix_for_two_crew_members)


def update_belief_matrix_for_two_crew_members(belief_matrix: np.array,
                                              ship_layout: list[list[str]],
                                              bot_position: tuple[int, int],
                                              alpha: float,
                                              is_beep: bool) -> np.ndarray:
    start_time = time.time()
    ship_dim = len(ship_layout)
    observation_matrix = get_observation_matrix_for_two_crew_members(ship_layout, bot_position, alpha, is_beep)
    updated_belief = belief_matrix * observation_matrix
    total_probability = updated_belief.sum()
    updated_belief = updated_belief/total_probability
    print(f'Time taken for updating 2 crew members belief:{time.time()-start_time}')
    return updated_belief
