import time

import numpy as np
from src.Utilities.utility import get_num_of_open_cells_outside_radius_k, get_open_neighbors


def initialize_belief_matrix_for_two_aliens(ship_layout: list[list[str]],
                                            bot_position: tuple[int, int], k: int) -> list[list[list[list[float]]]]:
    ship_dim = len(ship_layout)
    belief_matrix_for_two_aliens: list[list[list[list[float]]]] = [[[[
        0 for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)]
    n = get_num_of_open_cells_outside_radius_k(ship_layout, bot_position, k)

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    # Initialize the belief matrix for two aliens
    for i1 in range(ship_dim):
        for j1 in range(ship_dim):
            for i2 in range(ship_dim):
                for j2 in range(ship_dim):
                    if ((i1 < top or i1 > bottom or j1 < left or j1 > right) and (
                            i2 < top or i2 > bottom or j2 < left or j2 > right)
                            and ship_layout[i1][j1] != 'C' and ship_layout[i2][j2] != 'C'):
                        # As two aliens can't be in the same cell
                        belief_matrix_for_two_aliens[i1][j1][i2][j2] = 1 / (n * (n - 1))
    return belief_matrix_for_two_aliens


def get_observation_matrix_for_two_aliens(ship_layout: list[list[str]], bot_position: tuple[int, int],
                                          k: int, alien_sensed: bool) -> list[list[list[list[float]]]]:
    ship_dim = len(ship_layout)
    observation_matrix_for_two_aliens: list[list[list[list[float]]]] = [[[[
        0 for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)] for _ in range(ship_dim)]

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    for i1 in range(ship_dim):
        for j1 in range(ship_dim):
            for i2 in range(ship_dim):
                for j2 in range(ship_dim):
                    if alien_sensed:
                        # If the alien is sensed, set to 1 for open cells inside the 2k+1*2k+1 square
                        if (((top <= i1 <= bottom and left <= j1 <= right) or (
                                top <= i2 <= bottom and left <= j2 <= right))
                                and ship_layout[i1][j1] != 'C' and ship_layout[i2][j2] != 'C'):
                            observation_matrix_for_two_aliens[i1][j1][i2][j2] = 1
                    else:
                        # If the alien is not sensed, set to 1 for open cells outside the 2k+1*2k+1 square
                        if (((i1 < top or i1 > bottom or j1 < left or j1 > right) and (
                                i2 < top or i2 > bottom or j2 < left or j2 > right))
                                and ship_layout[i1][j1] != 'C' and ship_layout[i2][j2] != 'C'):
                            observation_matrix_for_two_aliens[i1][j1][i2][j2] = 1
    return observation_matrix_for_two_aliens


def get_transition_prob(open_neighbor_cells, ship_dim):
    # print(f'Ship dim received: {ship_dim}')
    transition_prob = np.zeros((ship_dim, ship_dim, ship_dim, ship_dim), float)
    print(transition_prob.shape)
    for i in range(ship_dim):
        for j in range(ship_dim):
            for k in range(ship_dim):
                for l in range(ship_dim):
                    if len(open_neighbor_cells[i][j]) != 0 and len(open_neighbor_cells[k][l]) != 0:
                        transition_prob[i][j][k][l] = 1 / (
                                    len(open_neighbor_cells[i][j]) * len(open_neighbor_cells[k][l]))
    return transition_prob


def update_belief_matrix_for_two_aliens(belief_matrix_for_two_aliens,
                                        ship_layout: list[list[str]],
                                        bot_position: tuple[int, int],
                                        k: int, alien_sensed: bool, transition_prob,
                                        open_neighbor_cells) -> list[list[list[list[float]]]]:
    ship_dim = len(ship_layout)
    start_time = time.time()
    observation_matrix_for_two_aliens = get_observation_matrix_for_two_aliens(ship_layout, bot_position, k,
                                                                              alien_sensed)
    print(f'Time taken for generating alien observation matrix is {time.time() - start_time}')
    updated_belief_matrix_for_two_aliens = np.zeros((ship_dim, ship_dim, ship_dim, ship_dim), float)
    belief_matrix_for_two_aliens = belief_matrix_for_two_aliens * transition_prob
    for i1 in range(ship_dim):
        for j1 in range(ship_dim):
            for i2 in range(ship_dim):
                for j2 in range(ship_dim):
                    if ship_layout[i1][j1] != 'C' and ship_layout[i2][j2] != 'C':
                        # Update belief based on the transition probabilities of neighboring cells
                        for neighbor1 in open_neighbor_cells[i1][j1]:
                            for neighbor2 in open_neighbor_cells[i2][j2]:
                                if neighbor1 != neighbor2:
                                    updated_belief_matrix_for_two_aliens[i1][j1][i2][j2] += (
                                            belief_matrix_for_two_aliens[neighbor1[0]][neighbor1[1]][neighbor2[0]][
                                                neighbor2[1]] *
                                            observation_matrix_for_two_aliens[i1][j1][i2][j2])
    updated_belief_matrix_for_two_aliens = updated_belief_matrix_for_two_aliens * observation_matrix_for_two_aliens
    # Normalize the updated belief matrix for two aliens
    total_probability = updated_belief_matrix_for_two_aliens.sum()
    updated_belief_matrix_for_two_aliens = updated_belief_matrix_for_two_aliens / total_probability
    return updated_belief_matrix_for_two_aliens
