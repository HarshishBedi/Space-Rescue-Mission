from src.Utilities.utility import get_num_of_open_cells_outside_radius_k, get_open_neighbors


def initialize_belief_matrix_for_one_alien(ship_layout: list[list[str]],
                                           bot_position: tuple[int, int], k: int) -> list[list[float]]:
    ship_dim = len(ship_layout)
    belief_matrix_for_one_alien: list[list[float]] = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]
    n = get_num_of_open_cells_outside_radius_k(ship_layout, bot_position, k)

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    # Initialize the belief matrix_for_one_alien
    for i in range(ship_dim):
        for j in range(ship_dim):
            if (i < top or i > bottom or j < left or j > right) and ship_layout[i][j] != 'C':
                belief_matrix_for_one_alien[i][j] = 1 / n
    return belief_matrix_for_one_alien


def get_observation_matrix_for_one_alien(ship_layout: list[list[str]],
                                         bot_position: tuple[int, int], k: int, alien_sensed: bool):
    ship_dim = len(ship_layout)
    observation_matrix_for_one_alien = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    # Set the observation matrix_for_one_alien values based on whether the alien is sensed
    for i in range(ship_dim):
        for j in range(ship_dim):
            if alien_sensed:
                # If the alien is sensed, set to 1 for open cells inside the 2k+1*2k+1 square
                if top <= i <= bottom and left <= j <= right and ship_layout[i][j] != 'C':
                    observation_matrix_for_one_alien[i][j] = 1
            else:
                # If the alien is not sensed, set to 1 for open cells outside the square
                if (i < top or i > bottom or j < left or j > right) and ship_layout[i][j] != 'C':
                    observation_matrix_for_one_alien[i][j] = 1
    return observation_matrix_for_one_alien


def update_belief_matrix_for_one_alien(belief_matrix_for_one_alien: list[list[float]],
                                       ship_layout: list[list[str]],
                                       bot_position: tuple[int, int],
                                       k: int, alien_sensed: bool) -> list[list[float]]:
    ship_dim = len(ship_layout)
    observation_matrix_for_one_alien = get_observation_matrix_for_one_alien(ship_layout, bot_position, k,
                                                                            alien_sensed)
    updated_belief_matrix_for_one_alien = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]
    for i in range(ship_dim):
        for j in range(ship_dim):
            # As transition probability is non-zero only for neighboring cells
            neighbors = get_open_neighbors((i, j), ship_layout)
            for neighbor in neighbors:
                num_of_neighbors = len(get_open_neighbors(neighbor, ship_layout))
                transition_prob = 1 / num_of_neighbors  # transition probability from neighbor to cell (i,j)
                updated_belief_matrix_for_one_alien[i][j] += (belief_matrix_for_one_alien[neighbor[0]][neighbor[1]] *
                                                              observation_matrix_for_one_alien[i][j] * transition_prob)
    # Normalize the updated belief matrix for one alien
    total_probability = sum(sum(row) for row in updated_belief_matrix_for_one_alien)
    for i in range(ship_dim):
        for j in range(ship_dim):
            updated_belief_matrix_for_one_alien[i][j] /= total_probability
    return updated_belief_matrix_for_one_alien
