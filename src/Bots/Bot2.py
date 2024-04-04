import time

import numpy as np
from collections import deque
from src.BeliefUpdates.Aliens.OneAlien import update_belief_matrix_for_one_alien
from src.BeliefUpdates.CrewMembers.OneCrewMember import update_belief_matrix_for_one_crew_member
from src.Utilities.Status import Status
from src.Utilities.utility import get_open_neighbors, calculate_information_gain


class Bot2:
    def __init__(self, bot_init_coords, alien_belief, crew_member_belief, alpha, k):
        self.position = bot_init_coords
        self.alien_belief = alien_belief
        self.crew_member_belief = crew_member_belief
        self.alpha = alpha
        self.k = k
        self.utility_weights = {'risk': 0.1, 'information_gain': 0.005, 'success': 0.4}
        self.goal = (-1, -1)
        self.path = []

    def update_beliefs(self, ship_layout, alien_beep, crew_member_beep):
        self.crew_member_belief = update_belief_matrix_for_one_crew_member(
            self.crew_member_belief, ship_layout, self.position, self.alpha, crew_member_beep)
        self.alien_belief = update_belief_matrix_for_one_alien(
            self.alien_belief, ship_layout, self.position, self.k, alien_beep)
        return self.crew_member_belief, self.alien_belief

    def calculate_utility(self,ship_layout):
        information_gain = calculate_information_gain(self.crew_member_belief, ship_layout, self.alpha)
        utility = (self.utility_weights['success'] * self.crew_member_belief -
                   self.utility_weights['risk'] * self.alien_belief +
                   self.utility_weights['information_gain'] * information_gain)
        return utility

    def bot_step(self, ship_layout):
        path = self.calculate_path(ship_layout)

        if path:
            self.path = path
            next_position = path[0]
            if ship_layout[next_position[0]][next_position[1]] == 'CM':
                ship_layout[self.position[0]][self.position[1]] = 'O'
                ship_layout[next_position[0]][next_position[1]] = 'CM&B'
                self.position = next_position
                return Status.SUCCESS, ship_layout, self.position, 1
            if (ship_layout[next_position[0]][next_position[1]] == 'A'
                    or ship_layout[next_position[0]][next_position[1]] == 'CM&A'):
                ship_layout[self.position[0]][self.position[1]] = 'O'
                ship_layout[next_position[0]][next_position[1]] = 'B&A'
                self.position = next_position
                return Status.FAILURE, ship_layout, self.position, 0
            ship_layout[self.position[0]][self.position[1]] = 'O'  # Clear the old position
            self.position = next_position
            ship_layout[self.position[0]][self.position[1]] = 'B'  # Mark the new position
            # Update the beliefs of alien and crew member positions:
            self.alien_belief[self.position[0]][self.position[1]] = 0.0
            self.crew_member_belief[self.position[0]][self.position[1]] = 0.0
        return Status.INPROCESS, ship_layout, self.position, 0

    def calculate_path(self, ship_layout):
        utility = self.calculate_utility(ship_layout)
        max_utility_posn = np.unravel_index(np.argmax(utility), utility.shape)
        self.goal = max_utility_posn
        fringe = deque([(self.position, deque())])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Directions: Up, Down, Left, Right
        visited = {self.position}  # Keep track of visited positions to avoid loops
        neighbors = get_open_neighbors(self.position, ship_layout)
        for neighbor in neighbors:
            if self.alien_belief[neighbor[0]][neighbor[1]] == 0:
                fringe.append((neighbor, deque([neighbor])))
                visited.add(neighbor)
        if len(fringe) > 1:
            fringe.popleft()
        while fringe:
            current_position, path = fringe.popleft()
            if current_position == max_utility_posn:
                return path
            for dx, dy in directions:
                nx, ny = current_position[0] + dx, current_position[1] + dy
                next_position = (nx, ny)
                if (0 <= nx < len(ship_layout) and 0 <= ny < len(ship_layout[0]) and next_position not in visited
                        and ship_layout[nx][ny] != 'C'):
                    visited.add(next_position)
                    new_path = path.copy()
                    new_path.append(next_position)
                    # Check if the next position is the goal
                    if next_position == max_utility_posn:
                        return new_path
                    fringe.append((next_position, new_path))
        return None
