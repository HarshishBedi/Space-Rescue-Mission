import numpy as np
from collections import deque
from src.BeliefUpdates.Aliens.TwoAliens import update_belief_matrix_for_two_aliens, get_transition_prob
from src.BeliefUpdates.CrewMembers.OneCrewMember import update_belief_matrix_for_one_crew_member
from src.BeliefUpdates.CrewMembers.TwoCrewMembers import update_belief_matrix_for_two_crew_members
from src.Utilities.Status import Status
from src.Utilities.utility import get_open_neighbors, calculate_information_gain, marginalize_belief, \
    k_largest_index_argpartition, get_mask_of_k_max


class Bot8:
    def __init__(self, bot_init_coords, alien_belief, crew_member_belief, alpha, k, number_of_crew_members: int = 2,
                 open_neighbor_cells=[]):
        if open_neighbor_cells is None:
            open_neighbor_cells = []
        self.position = bot_init_coords
        self.alien_belief = alien_belief
        self.crew_member_belief = crew_member_belief
        self.alpha = alpha
        self.k = k
        self.utility_weights = {'risk': 0.1, 'information_gain': 0.025, 'success': 0.4}
        self.goal = (-1, -1)
        self.path = []
        self.num_of_crew_members_saved = 0
        self.number_of_crew_members = number_of_crew_members
        self.open_neighbor_cells = open_neighbor_cells
        self.transition_prob = get_transition_prob(open_neighbor_cells)

    def update_beliefs(self, ship_layout, alien_beep, crew_member_beep):
        if self.num_of_crew_members_saved == 0:
            self.crew_member_belief = update_belief_matrix_for_two_crew_members(self.crew_member_belief, ship_layout,
                                                                                self.position, self.alpha,
                                                                                crew_member_beep)
        else:
            self.crew_member_belief = update_belief_matrix_for_one_crew_member(self.crew_member_belief, ship_layout,
                                                                               self.position, self.alpha,
                                                                               crew_member_beep)
        self.alien_belief = update_belief_matrix_for_two_aliens(self.alien_belief, ship_layout,
                                                                self.position,
                                                                self.k, alien_beep,
                                                                transition_prob=self.transition_prob)
        return self.crew_member_belief, self.alien_belief

    def calculate_utility(self, ship_layout):
        ship_dim = len(ship_layout)
        k_max_crew_mate_belief_mask = get_mask_of_k_max(ship_dim, k_largest_index_argpartition(self.crew_member_belief,
                                                                                               2),
                                                        (self.number_of_crew_members - self.num_of_crew_members_saved))
        information_gain = calculate_information_gain(self.crew_member_belief, ship_layout, self.alpha,
                                                      self.number_of_crew_members - self.num_of_crew_members_saved,
                                                      open_cell_mask=k_max_crew_mate_belief_mask)
        if self.num_of_crew_members_saved==0:
            utility = (self.utility_weights['success'] * (marginalize_belief(self.crew_member_belief, (2, 3)) +
                                                          marginalize_belief(self.crew_member_belief, (0, 1))) -
                       self.utility_weights['risk'] * (marginalize_belief(self.alien_belief, (2, 3)) +
                                                       marginalize_belief(self.alien_belief, (0, 1))) +
                       self.utility_weights['information_gain'] * information_gain)
        else:
            utility = (self.utility_weights['success'] * self.crew_member_belief -
                       self.utility_weights['risk'] * (marginalize_belief(self.alien_belief, (2, 3)) +
                                                       marginalize_belief(self.alien_belief, (0, 1))) +
                       self.utility_weights['information_gain'] * information_gain)
        return utility

    def bot_step(self, ship_layout):
        path = self.calculate_path(ship_layout)
        if path:
            self.path = path
            next_position = path[0]
            if ship_layout[next_position[0]][next_position[1]] == 'CM':
                self.num_of_crew_members_saved += 1
                if self.num_of_crew_members_saved == self.number_of_crew_members:
                    ship_layout[self.position[0]][self.position[1]] = 'O'
                    ship_layout[next_position[0]][next_position[1]] = 'CM&B'
                    self.position = next_position
                    return Status.SUCCESS, ship_layout, self.position, self.num_of_crew_members_saved
                self.crew_member_belief = marginalize_belief(self.crew_member_belief, (2, 3))
            if (ship_layout[next_position[0]][next_position[1]] == 'A'
                    or ship_layout[next_position[0]][next_position[1]] == 'CM&A'
                    or ship_layout[next_position[0]][next_position[1]] == 'CM&A&A'):
                ship_layout[self.position[0]][self.position[1]] = 'O'
                ship_layout[next_position[0]][next_position[1]] = 'B&A'
                self.position = next_position
                return Status.FAILURE, ship_layout, self.position, self.num_of_crew_members_saved
            ship_layout[self.position[0]][self.position[1]] = 'O'  # Clear the old position
            self.position = next_position
            ship_layout[self.position[0]][self.position[1]] = 'B'  # Mark the new position
            # Update the beliefs of alien and crew member positions:
            self.update_belief_based_on_bot_step(ship_layout)
        return Status.INPROCESS, ship_layout, self.position, self.num_of_crew_members_saved

    def calculate_path(self, ship_layout):
        utility = self.calculate_utility(ship_layout)
        max_utility_posn = np.unravel_index(np.argmax(utility), utility.shape)
        self.goal = max_utility_posn
        fringe = deque([(self.position, deque())])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Directions: Up, Down, Left, Right
        visited = {self.position}  # Keep track of visited positions to avoid loops
        neighbors = get_open_neighbors(self.position, ship_layout)
        for neighbor in neighbors:
            if self.is_cell_alien_safe(neighbor):
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

    def update_belief_based_on_bot_step(self, ship_layout):
        ship_dim = len(ship_layout)
        self.alien_belief[self.position[0]][self.position[1]] = 0.0
        for i in range(ship_dim):
            for j in range(ship_dim):
                self.alien_belief[i][j][self.position[0]][self.position[1]] = 0.0
                self.alien_belief[self.position[0]][self.position[1]][i][j] = 0.0
                if self.num_of_crew_members_saved == 0:
                    self.crew_member_belief[i][j][self.position[0]][self.position[1]] = 0.0
                    self.crew_member_belief[self.position[0]][self.position[1]][i][j] = 0.0
        if self.num_of_crew_members_saved == 1:
            self.crew_member_belief[self.position[0]][self.position[1]] = 0.0

    def is_cell_alien_safe(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        alien_chance_in_cell = np.sum(self.alien_belief[:, :, x, y] + self.alien_belief[x, y, :, :])
        return alien_chance_in_cell == 0.0
