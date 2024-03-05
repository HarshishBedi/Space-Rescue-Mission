import random
from Status import Status


def alien_step(ship_layout: list[list[str]], aliens: list[tuple[int, int]]) -> tuple[Status, list[list[str]], list[tuple[int, int]]]:
    """
    :param ship_layout:layout of the ship as a 2D matrix with each element representing whether the cell at that
                        coordinates is open/closed/occupied by someone(Eg: Alien/Bot/Captain)
    :param aliens: set containing the positions of aliens.
    :return: returns a tuple containing 3 elements:
                1. Status of the simulation, which indicates whether the bot failed/succeeded/ or in process of finding
                   the captain
                2. Updated layout of the ship as a 2D matrix, after all aliens make a move randomly.
                3. Set containing the updated positions of the aliens.
    """
    # directions for the neighboring cell calculation
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # left, right, up, down
    # randomizing the order in which the aliens move
    random.shuffle(aliens)
    # rules for updating the neighboring cell into which the alien will move into
    # if the neighboring cell contains the captain(CP), the updated cell will contain both captain and the alien (CP&A)
    # after the step. If the neighboring cells is an unoccupied open cell, then the cell will be updated to contain the
    # alien(A) after the step
    rules_for_updating_new_alien_square = {'CP': 'CP&A', 'O': 'A'}
    # looping through the randomized set of aliens
    for i in range(len(aliens)):
        # alien coordinates
        alien_x, alien_y = aliens[i]
        possible_steps = []
        # identifying the valid neighboring cells for the alien based on the following criteria:
        #   1. The cell should not be closed
        #   2. The cell should not contain another alien
        for dx, dy in directions:
            nx, ny = alien_x + dx, alien_y + dy
            if 0 <= nx < len(ship_layout) and 0 <= ny < len(ship_layout[0]):
                if ship_layout[nx][ny] != 'C' and ship_layout[nx][ny] != 'A' and ship_layout[nx][ny] != 'CP&A':
                    possible_steps.append((nx, ny))
        # If no valid neighboring cells found based on the criteria mentioned above, then the alien stays in its current
        # position.
        if not possible_steps:
            continue
        # Randomly selecting a cell from the list of valid neighboring cells, for the alien to move into
        nx, ny = random.choice(possible_steps)
        # Updating the alien's position within the list of alien positions, to the randomly selected cell from the list
        # of valid neighboring cells for the aliens.
        aliens[i] = (nx, ny)
        # Updating the current square of alien within the ship layout to remove alien
        if ship_layout[alien_x][alien_y] == 'CP&A':
            # If the current position of alien contains both captain and alien('CP&A'), it will be updated to contain
            # only the captain
            ship_layout[alien_x][alien_y] = 'CP'
        elif ship_layout[alien_x][alien_y] == 'A':
            # If the current position of the alien contains only the alien ('A'), then it will be updated to an
            #  unoccupied open cell('O')
            ship_layout[alien_x][alien_y] = 'O'
        # Updating the randomly selected next square of alien within the ship layout to place the alien
        if ship_layout[nx][ny] == 'B':
            # If the next cell of the alien contains the bot, then the task fails. So we return the status of
            # the simulation as FAILURE
            ship_layout[nx][ny] = 'B&A'
            return Status.FAILURE, ship_layout, aliens
        else:
            # Using the rules dictionary defined earlier, to update the next position of the alien, within
            # the ship layout
            ship_layout[nx][ny] = rules_for_updating_new_alien_square[ship_layout[nx][ny]]

    # If the aliens encounter the bot at any point, then the task fails, we return Failure status. So, if the code
    # reaches this point, it means that none of the aliens encountered the bot in the current iteration, so we return
    # the status of the simulation as INPROCESS. Along with the status we return the updated ship layout and the updated
    # positions of the bot.
    return Status.INPROCESS, ship_layout, aliens
