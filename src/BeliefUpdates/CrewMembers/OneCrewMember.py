import math

from src.Utilities.utility import get_num_of_open_cells_in_ship, get_manhattan_distance


def initialize_belief_matrix(ship_layout: list[list[str]]) -> list[list[float]]:
    n = get_num_of_open_cells_in_ship(ship_layout)
    belief_matrix: list[list[float]] = [[0 for _ in range(len(ship_layout))] for _ in range(len(ship_layout))]
    for i in range(len(ship_layout)):
        for j in range(len(ship_layout)):
            if ship_layout[i][j] in ['C', 'B']:
                belief_matrix[i][j] = 0
            else:
                belief_matrix[i][j] = 1 / (n - 1)
    return belief_matrix


def get_observation_matrix(ship_layout: list[list[str]], bot_position: tuple[int, int], alpha: int, is_beep: bool) -> list[list[float]]:
    ship_dim = len(ship_layout)
    observation_matrix: list[list[float]] = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]
    for i in range(ship_dim):
        for j in range(ship_dim):
            d = get_manhattan_distance(bot_position, (i,j))
            if is_beep:
                observation_matrix[i][j] = math.exp(-alpha * (d - 1))
            else:
                observation_matrix[i][j] = 1 - math.exp(-alpha * (d - 1))
    return observation_matrix


def update_belief_matrix(belief_matrix: list[list[float]], ship_layout: list[list[str]],
                         bot_position: tuple[int, int], alpha: int, is_beep: bool) -> list[list[float]]:
    ship_dim = len(ship_layout)
    observation_matrix = get_observation_matrix(ship_layout, bot_position, alpha, is_beep)

    updated_belief_matrix = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]
    for i in range(ship_dim):
        for j in range(ship_dim):
            if ship_layout[i][j] == 'O':
                updated_belief_matrix[i][j] = belief_matrix[i][j] * observation_matrix[i][j]  # As transition model is 1
            else:
                updated_belief_matrix[i][j] = 0

    # Normalize the updated belief matrix
    total_probability = sum(sum(row) for row in updated_belief_matrix)
    for i in range(ship_dim):
        for j in range(ship_dim):
            updated_belief_matrix[i][j] /= total_probability
    return updated_belief_matrix

