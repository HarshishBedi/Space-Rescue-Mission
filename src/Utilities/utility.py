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