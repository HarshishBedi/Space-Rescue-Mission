import time

import numpy as np
from src.Utilities.utility import get_num_of_open_cells_outside_radius_k, get_open_neighbors


def initialize_belief_matrix_for_two_aliens(ship_layout: list[list[str]],
                                            bot_position: tuple[int, int], k: int) -> np.ndarray:
    ship_dim = len(ship_layout)
    belief_matrix_for_two_aliens = np.zeros((ship_dim, ship_dim, ship_dim, ship_dim), float)
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
                                          k: int, alien_sensed: bool) -> np.ndarray:
    ship_dim = len(ship_layout)
    observation_matrix_for_two_aliens = np.zeros((ship_dim, ship_dim, ship_dim, ship_dim), float)

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


def get_transition_prob(open_neighbor_cells):
    # Convert the list of open neighbor cells to a 2D array of counts
    open_neighbor_counts = np.array([[len(neighbors) for neighbors in row] for row in open_neighbor_cells], dtype=float)

    # Avoid division by zero by setting zero counts to infinity (so that 1 / count will be zero)
    open_neighbor_counts[open_neighbor_counts == 0] = np.inf

    # Calculate the transition probabilities using broadcasting
    transition_prob = 1 / (open_neighbor_counts[:, :, None, None] * open_neighbor_counts[None, None, :, :])
    return transition_prob


def update_belief_matrix_for_two_aliens(belief_matrix_for_two_aliens: np.ndarray,
                                        ship_layout: list[list[str]],
                                        bot_position: tuple[int, int],
                                        k: int, alien_sensed: bool, transition_prob,
                                        open_neighbor_cells) -> np.ndarray:
    start = time.time()
    ship_dim = len(ship_layout)
    observation_matrix_for_two_aliens = get_observation_matrix_for_two_aliens(ship_layout, bot_position, k,
                                                                              alien_sensed)
    print(f'Time taken for getting observation matrix is {time.time() - start}')
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
                                # if neighbor1 != neighbor2:
                                updated_belief_matrix_for_two_aliens[i1][j1][i2][j2] += (
                                    belief_matrix_for_two_aliens[neighbor1[0]][neighbor1[1]][neighbor2[0]][
                                        neighbor2[1]]) * observation_matrix_for_two_aliens[i1][j1][i2][j2]
    # updated_belief_matrix_for_two_aliens = updated_belief_matrix_for_two_aliens * observation_matrix_for_two_aliens
    # Normalize the updated belief matrix for two aliens
    # total_probability = updated_belief_matrix_for_two_aliens.sum()
    # updated_belief_matrix_for_two_aliens = updated_belief_matrix_for_two_aliens / total_probability
    print(f"Time taken for updating belief: {time.time() - start}")
    return updated_belief_matrix_for_two_aliens


def update_belief_matrix_for_two_aliens_vectorized(belief_matrix_for_two_aliens: np.ndarray,
                                                   ship_layout: list[list[str]],
                                                   bot_position: tuple[int, int],
                                                   k: int, alien_sensed: bool, transition_prob) -> np.ndarray:
    start = time.time()
    ship_dim = len(ship_layout)
    observation_matrix_for_two_aliens = get_observation_matrix_for_two_aliens(ship_layout, bot_position, k,
                                                                              alien_sensed)
    # print(f'Time taken for getting observation matrix is {time.time() - start}')
    belief_matrix_for_two_aliens = belief_matrix_for_two_aliens * transition_prob
    updated_belief_matrix_for_two_aliens = add_neighbors(belief_matrix_for_two_aliens)
    updated_belief_matrix_for_two_aliens = updated_belief_matrix_for_two_aliens * observation_matrix_for_two_aliens
    for i in range(ship_dim):
        for j in range(ship_dim):
            if ship_layout[i][j] == 'C':
                updated_belief_matrix_for_two_aliens[i][j] = 0
    # Normalize the updated belief matrix for two aliens
    # total_probability = updated_belief_matrix_for_two_aliens.sum()
    # updated_belief_matrix_for_two_aliens = updated_belief_matrix_for_two_aliens / total_probability
    print(f"Time taken for updating belief: {time.time() - start}")
    return updated_belief_matrix_for_two_aliens


def add_neighbors(belief_matrix):
    ################################
    ship_dim = len(belief_matrix)
    updated_belief_matrix1 = np.zeros((ship_dim, ship_dim, ship_dim, ship_dim), float)
    shifted_belief = np.concatenate((belief_matrix, np.zeros((1, ship_dim, ship_dim, ship_dim), float)), 0)
    shifted_belief = shifted_belief[1:, :, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, ship_dim, 1), float), shifted_belief), 3)
    shifted_belief = shifted_belief[:, :, :, :-1]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((belief_matrix, np.zeros((1, ship_dim, ship_dim, ship_dim), float)), 0)
    shifted_belief = shifted_belief[1:, :, :, :]
    shifted_belief = np.concatenate((shifted_belief,np.zeros((ship_dim, ship_dim, ship_dim, 1), float)), 3)
    shifted_belief = shifted_belief[:, :, :, 1:]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((belief_matrix, np.zeros((1, ship_dim, ship_dim, ship_dim), float)), 0)
    shifted_belief = shifted_belief[1:, :, :, :]
    shifted_belief = np.concatenate((shifted_belief,np.zeros((ship_dim, ship_dim, 1, ship_dim), float)), 2)
    shifted_belief = shifted_belief[:, :, 1:, :]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((belief_matrix, np.zeros((1, ship_dim, ship_dim, ship_dim), float)), 0)
    shifted_belief = shifted_belief[1:, :, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, 1, ship_dim), float),shifted_belief), 2)
    shifted_belief = shifted_belief[:, :, :-1, :]
    updated_belief_matrix1 += shifted_belief
    ##############################################
    shifted_belief = np.concatenate((np.zeros((1, ship_dim, ship_dim, ship_dim), float),belief_matrix), 0)
    shifted_belief = shifted_belief[:-1, :, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, ship_dim, 1), float), shifted_belief), 3)
    shifted_belief = shifted_belief[:, :, :, :-1]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((np.zeros((1, ship_dim, ship_dim, ship_dim), float),belief_matrix), 0)
    shifted_belief = shifted_belief[:-1, :, :, :]
    shifted_belief = np.concatenate((shifted_belief,np.zeros((ship_dim, ship_dim, ship_dim, 1), float)), 3)
    shifted_belief = shifted_belief[:, :, :, 1:]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((np.zeros((1, ship_dim, ship_dim, ship_dim), float),belief_matrix), 0)
    shifted_belief = shifted_belief[:-1, :, :, :]
    shifted_belief = np.concatenate((shifted_belief,np.zeros((ship_dim, ship_dim, 1, ship_dim), float)), 2)
    shifted_belief = shifted_belief[:, :, 1:, :]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((np.zeros((1, ship_dim, ship_dim, ship_dim), float),belief_matrix), 0)
    shifted_belief = shifted_belief[:-1, :, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, 1, ship_dim), float),shifted_belief), 2)
    shifted_belief = shifted_belief[:, :, :-1, :]
    updated_belief_matrix1 += shifted_belief
    #########################################
    shifted_belief = np.concatenate((np.zeros((ship_dim, 1, ship_dim, ship_dim), float), belief_matrix), 1)
    shifted_belief = shifted_belief[:, :-1, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, ship_dim, 1), float), shifted_belief), 3)
    shifted_belief = shifted_belief[:, :, :, :-1]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((np.zeros((ship_dim, 1, ship_dim, ship_dim), float), belief_matrix), 1)
    shifted_belief = shifted_belief[:, :-1, :, :]
    shifted_belief = np.concatenate((shifted_belief, np.zeros((ship_dim, ship_dim, ship_dim, 1), float)), 3)
    shifted_belief = shifted_belief[:, :, :, 1:]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((np.zeros((ship_dim, 1, ship_dim, ship_dim), float), belief_matrix), 1)
    shifted_belief = shifted_belief[:, :-1, :, :]
    shifted_belief = np.concatenate((shifted_belief, np.zeros((ship_dim, ship_dim, 1, ship_dim), float)), 2)
    shifted_belief = shifted_belief[:, :, 1:, :]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((np.zeros((ship_dim, 1, ship_dim, ship_dim), float), belief_matrix), 1)
    shifted_belief = shifted_belief[:, :-1, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, 1, ship_dim), float), shifted_belief), 2)
    shifted_belief = shifted_belief[:, :, :-1, :]
    updated_belief_matrix1 += shifted_belief
    ################################################
    shifted_belief = np.concatenate((belief_matrix,np.zeros((ship_dim, 1, ship_dim, ship_dim), float)), 1)
    shifted_belief = shifted_belief[:, 1:, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, ship_dim, 1), float), shifted_belief), 3)
    shifted_belief = shifted_belief[:, :, :, :-1]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((belief_matrix,np.zeros((ship_dim, 1, ship_dim, ship_dim), float)), 1)
    shifted_belief = shifted_belief[:, 1:, :, :]
    shifted_belief = np.concatenate((shifted_belief, np.zeros((ship_dim, ship_dim, ship_dim, 1), float)), 3)
    shifted_belief = shifted_belief[:, :, :, 1:]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((belief_matrix,np.zeros((ship_dim, 1, ship_dim, ship_dim), float)), 1)
    shifted_belief = shifted_belief[:, 1:, :, :]
    shifted_belief = np.concatenate((shifted_belief, np.zeros((ship_dim, ship_dim, 1, ship_dim), float)), 2)
    shifted_belief = shifted_belief[:, :, 1:, :]
    updated_belief_matrix1 += shifted_belief
    shifted_belief = np.concatenate((belief_matrix,np.zeros((ship_dim, 1, ship_dim, ship_dim), float)), 1)
    shifted_belief = shifted_belief[:, 1:, :, :]
    shifted_belief = np.concatenate((np.zeros((ship_dim, ship_dim, 1, ship_dim), float), shifted_belief), 2)
    shifted_belief = shifted_belief[:, :, :-1, :]
    updated_belief_matrix1 += shifted_belief
    return updated_belief_matrix1
