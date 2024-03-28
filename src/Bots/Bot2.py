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
        self.utility_weights = {'proximity': 0.4, 'risk': 0.35, 'information_gain': 0.15, 'success': 0.4}

    def update_beliefs(self, ship_layout, alien_beep, crew_member_beep):
        self.crew_member_belief = update_belief_matrix_for_one_crew_member(
            self.crew_member_belief, ship_layout, self.position, self.alpha, crew_member_beep)
        self.alien_belief = update_belief_matrix_for_one_alien(
            self.alien_belief, ship_layout, self.position, self.k, alien_beep)
        return self.crew_member_belief, self.alien_belief

    def calculate_utility(self, new_position, ship_layout):
        proximity = -np.linalg.norm(np.array(new_position) - np.array(self.get_max_belief_crew_member_position()))
        risk = -self.alien_belief[new_position[0], new_position[1]]
        information_gain = calculate_information_gain(self.crew_member_belief, self.position, new_position)
        success = self.crew_member_belief[new_position[0], new_position[1]]
        utility = (self.utility_weights['proximity'] * proximity +
                   self.utility_weights['risk'] * risk +
                   self.utility_weights['information_gain'] * information_gain +
                   self.utility_weights['success'] * success)
        return utility

    def get_max_belief_crew_member_position(self):
        return np.unravel_index(np.argmax(self.crew_member_belief), self.crew_member_belief.shape)

    def bot_step(self, ship_layout):
        neighbors = get_open_neighbors(self.position, ship_layout)
        best_neighbor = None
        max_utility = float('-inf')
        for neighbor in neighbors:
            utility = self.calculate_utility(neighbor, ship_layout)
            if utility > max_utility:
                max_utility = utility
                best_neighbor = neighbor

        if best_neighbor:
            self.position = best_neighbor
            # Update the ship layout
            ship_layout[self.position[0]][self.position[1]] = 'B'
            if ship_layout[best_neighbor[0]][best_neighbor[1]] == 'CM':
                return Status.SUCCESS, ship_layout, self.position, 1
            elif ship_layout[best_neighbor[0]][best_neighbor[1]] in ['A', 'CM&A']:
                return Status.FAILURE, ship_layout, self.position, 0
        return Status.INPROCESS, ship_layout, self.position, 0