from collections import deque
import numpy as np
from src.BeliefUpdates.Aliens.OneAlien import update_belief_matrix_for_one_alien
from src.BeliefUpdates.CrewMembers.OneCrewMember import update_belief_matrix_for_one_crew_member
from src.Utilities.Status import Status
from src.Utilities.utility import get_open_neighbors


class Bot1:
    def __init__(self, bot_init_coords: tuple[int, int], alien_belief: list[list[float]],
                 crew_member_belief: list[list[float]], alpha: float, k: int):
        self.position = bot_init_coords
        self.alien_belief = alien_belief
        self.crew_member_belief = crew_member_belief
        self.alpha = alpha
        self.k = k
        self.goal = (-1, -1)
        self.path = None

    def update_beliefs(self, ship_layout: list[list[str]], alien_beep: bool, crew_member_beep: bool):
        self.crew_member_belief = update_belief_matrix_for_one_crew_member(self.crew_member_belief, ship_layout,
                                                                           self.position, self.alpha, crew_member_beep)
        self.alien_belief = update_belief_matrix_for_one_alien(self.alien_belief, ship_layout, self.position,
                                                               self.k, alien_beep)
        return self.crew_member_belief, self.alien_belief

    def get_max_belief_crew_member_position(self):
        return np.unravel_index(np.argmax(self.crew_member_belief), self.crew_member_belief.shape)

    def calculate_path(self, ship_layout):
        goal_position = self.get_max_belief_crew_member_position()
        self.goal = goal_position
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
            if current_position == goal_position:
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
                    if next_position == goal_position:
                        return new_path
                    fringe.append((next_position, new_path))
        return None

    def bot_step(self, ship_layout):
        path = self.calculate_path(ship_layout)
        self.path = path
        if path:
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
