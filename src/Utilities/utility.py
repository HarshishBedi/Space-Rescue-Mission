import itertools

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


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

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args)

def export_to_pdf(data):
    print('Writing to PDF')
    c = canvas.Canvas("grid-students.pdf", pagesize=A4)
    w, h = A4
    max_rows_per_page = 45
    # Margin.
    x_offset = 50
    y_offset = 50
    # Space between rows.
    padding = 15

    xlist = [x + x_offset for x in [0, 200, 250, 300, 350, 400]]
    ylist = [h - y_offset - i * padding for i in range(max_rows_per_page + 1)]

    for rows in grouper(data, max_rows_per_page):
        rows = tuple(filter(bool, rows))
        c.grid(xlist, ylist[:len(rows) + 1])
        for y, row in zip(ylist[:-1], rows):
            for x, cell in zip(xlist, row):
                c.drawString(x + 2, y - padding + 3, str(cell))
        c.showPage()

    c.save()
