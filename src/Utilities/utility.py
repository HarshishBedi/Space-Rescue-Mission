import numpy as np
from fpdf import FPDF


def get_num_of_open_cells_in_ship(ship_layout: list[list[str]]):
    return sum(ship_layout[i][j] != 'C' for i in range(len(ship_layout)) for j in range(len(ship_layout[i])))


def get_num_of_open_cells_outside_radius_k(ship_layout, bot_position, k):
    num_open_cells = 0
    size = len(ship_layout)

    # Define the boundaries of the square centered at the bot's position
    top = max(0, bot_position[0] - k)
    bottom = min(size - 1, bot_position[0] + k)
    left = max(0, bot_position[1] - k)
    right = min(size - 1, bot_position[1] + k)

    # Iterate through the ship layout
    for i in range(size):
        for j in range(size):
            # Check if the cell is outside the square and is equal to 'O'
            if (i < top or i > bottom or j < left or j > right) and ship_layout[i][j] != 'C':
                num_open_cells += 1

    return num_open_cells


def get_manhattan_distance(bot_position, cell_position):
    return abs(bot_position[0] - cell_position[0]) + abs(bot_position[1] - cell_position[1])


def get_open_neighbors(position: tuple[int, int], ship_layout: list[list[str]]) -> list[tuple[int, int]]:
    # Define the possible directions (up, down, left, right)
    ship_dim = len(ship_layout)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors: list[tuple[int, int]] = []
    for dx, dy in directions:
        neighbor_x, neighbor_y = position[0] + dx, position[1] + dy
        if 0 <= neighbor_x < ship_dim and 0 <= neighbor_y < ship_dim and ship_layout[neighbor_x][neighbor_y] != 'C':
            neighbors.append((neighbor_x, neighbor_y))
    return neighbors


def open_neighbor_cells_matrix(ship_layout):
    ship_dim = len(ship_layout)
    open_neighbor_cells = [[[] for i in range(ship_dim)] for j in range(ship_dim)]
    for i in range(ship_dim):
        for j in range(ship_dim):
            neighbors = get_open_neighbors((i,j),ship_layout)
            open_neighbor_cells[i][j] = neighbors
    return open_neighbor_cells


def append_to_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=16)
    with pdf.table() as table:
        for data_row in data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.output('table.pdf')


def calculate_information_gain(belief_matrix, current_position, new_position):
    """
    Calculate the expected information gain from moving to the new position
    based on the reduction in entropy of the belief matrix.
    """
    # Calculate the entropy of the current and new positions
    current_entropy = entropy(belief_matrix[current_position[0], current_position[1]])
    new_entropy = entropy(belief_matrix[new_position[0], new_position[1]])

    # Information gain is the reduction in entropy
    info_gain = current_entropy - new_entropy

    return info_gain


def entropy(probability):
    """
    Calculate the entropy of a probability distribution.
    """
    if probability == 0 or probability == 1:
        return 0
    return -probability * np.log2(probability) - (1 - probability) * np.log2(1 - probability)
