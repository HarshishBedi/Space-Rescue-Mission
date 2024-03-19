import math
from random import random


def get_num_of_open_cells_in_crop(ship_layout:list[list[int]],center_cell:tuple[int,int] = (-1,-1),radius:int = -1):
    i_range = (-1,-1)
    j_range = (-1,-1)
    ship_dim = len(ship_layout)
    if center_cell == (-1,-1):
        i_range = (0,ship_dim)
        j_range = (0,ship_dim)
    num_of_open_cells = 0
    for i in range(i_range[0],i_range[1],1):
        for j in range(j_range[0]):
            if ship_layout[i][j] == 'O':
                num_of_open_cells += 1
    return num_of_open_cells


def get_num_of_open_cells_in_ship(ship_layout: list[list[str]]):
    return sum(ship_layout[i][j] == 'O' for i in range(len(ship_layout)) for j in range(len(ship_layout[i])))


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
            if (i < top or i > bottom or j < left or j > right) and ship_layout[i][j] == 'O':
                num_open_cells += 1

    return num_open_cells


def get_manhattan_distance(bot_position, cell_position):
    return abs(bot_position[0] - cell_position[0]) + abs(bot_position[1] - cell_position[1])

