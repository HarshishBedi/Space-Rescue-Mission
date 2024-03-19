from src.Utilities.utility import  get_num_of_open_cells_outside_radius_k


def initialize_belief_matrix(ship_layout: list[list[str]], bot_position: tuple[int, int], k: int) -> list[list[float]]:
    ship_dim = len(ship_layout)
    belief_matrix: list[list[float]] = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]
    n = get_num_of_open_cells_outside_radius_k(ship_layout, bot_position, k)

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    # Initialize the belief matrix
    for i in range(ship_dim):
        for j in range(ship_dim):
            if i < top or i > bottom or j < left or j > right:
                belief_matrix[i][j] = 1 / n
    return belief_matrix


def get_observation_matrix(ship_layout: list[list[str]], bot_position: tuple[int, int], k: int, alien_sensed: bool):
    ship_dim = len(ship_layout)
    observation_matrix = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(ship_dim - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(ship_dim - 1, bot_position[1] + k)

    # Set the observation matrix values based on whether the alien is sensed
    for i in range(ship_dim):
        for j in range(ship_dim):
            if alien_sensed:
                # If the alien is sensed, set to 1 for open cells inside the 2k+1*2k+1 square
                if top <= i <= bottom and left <= j <= right and ship_layout[i][j] == 'O':
                    observation_matrix[i][j] = 1
            else:
                # If the alien is not sensed, set to 1 for open cells outside the square
                if (i < top or i > bottom or j < left or j > right) and ship_layout[i][j] == 'O':
                    observation_matrix[i][j] = 1
    return observation_matrix


def get_transition_matrix(ship_layout: list[list[str]]) -> list[list[float]]:
    ship_dim = len(ship_layout)
    transition_matrix: list[list[float]] = [[0 for _ in range(ship_dim)] for _ in range(ship_dim)]

    # Define the possible directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(ship_dim):
        for j in range(ship_dim):
            if ship_layout[i][j] == 'O':
                open_neighbors = 0
                # Count the number of open neighboring cells
                for dx, dy in directions:
                    new_i, new_j = i + dx, j + dy
                    if 0 <= new_i < ship_dim and 0 <= new_j < ship_dim and ship_layout[new_i][new_j] == 'O':
                        open_neighbors += 1

                # Assign probabilities to the neighboring cells
                for dx, dy in directions:
                    new_i, new_j = i + dx, j + dy
                    if 0 <= new_i < ship_dim and 0 <= new_j < ship_dim and ship_layout[new_i][new_j] == 'O':
                        transition_matrix[new_i][new_j] = 1 / open_neighbors

    return transition_matrix
