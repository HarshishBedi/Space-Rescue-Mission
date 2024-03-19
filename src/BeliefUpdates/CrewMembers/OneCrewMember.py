import math

from src.Utilities.utility import get_num_of_open_cells_in_ship, get_manhattan_distance


def initialize_belief_matrix_for_one_crewmate(ship_layout: list[list[str]]) -> list[list[float]]:
    n = get_num_of_open_cells_in_ship(ship_layout)
    belief_matrix_for_one_crewmate: list[list[float]] = [[0 for _ in range(len(ship_layout))] for _ in
                                                         range(len(ship_layout))]
    for i in range(len(ship_layout)):
        for j in range(len(ship_layout)):
            if ship_layout[i][j] not in ['C', 'B']:
                belief_matrix_for_one_crewmate[i][j] = 1 / (n - 1)
    return belief_matrix_for_one_crewmate


def get_observation_matrix_for_one_crewmate(ship_layout: list[list[str]], bot_position: tuple[int, int], alpha: int,
                                            is_beep: bool) -> list[list[float]]:
    ship_dim = len(ship_layout)
    observation_matrix_for_one_crewmate: list[list[float]] = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]
    for i in range(ship_dim):
        for j in range(ship_dim):
            d = get_manhattan_distance(bot_position, (i, j))
            if is_beep:
                observation_matrix_for_one_crewmate[i][j] = math.exp(-alpha * (d - 1))
            else:
                observation_matrix_for_one_crewmate[i][j] = 1 - math.exp(-alpha * (d - 1))
    return observation_matrix_for_one_crewmate


def update_belief_matrix_for_one_crewmate(belief_matrix_for_one_crewmate: list[list[float]],
                                          ship_layout: list[list[str]],
                                          bot_position: tuple[int, int],
                                          alpha: float, is_beep: bool) -> list[list[float]]:
    ship_dim = len(ship_layout)
    observation_matrix_for_one_crewmate = get_observation_matrix_for_one_crewmate(ship_layout, bot_position, alpha,
                                                                                  is_beep)
    updated_belief_matrix_for_one_crewmate = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]
    for i in range(ship_dim):
        for j in range(ship_dim):
            # As transition model is 1
            updated_belief_matrix_for_one_crewmate[i][j] = belief_matrix_for_one_crewmate[i][j] * \
                                                           observation_matrix_for_one_crewmate[i][j]
    # Normalize the updated belief matrix_for_one_crewmate
    total_probability = sum(sum(row) for row in updated_belief_matrix_for_one_crewmate)
    for i in range(ship_dim):
        for j in range(ship_dim):
            updated_belief_matrix_for_one_crewmate[i][j] /= total_probability
    return updated_belief_matrix_for_one_crewmate
