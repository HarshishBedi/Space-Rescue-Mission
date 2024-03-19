import logging
from tkinter import Tk, ttk
import matplotlib.pyplot as plt

from src.BeliefUpdates.Aliens.OneAlien import initialize_belief_matrix_for_one_alien
from src.BeliefUpdates.CrewMembers.OneCrewMember import initialize_belief_matrix_for_one_crewmate
from src.Utilities.Ship import Ship
from src.Utilities.Spawner import Spawner


def show_tkinter(ship_layout: list[list[str]]):
    """
    :param ship_layout: layout of the ship as a 2D matrix with each element representing whether the cell at that
                        coordinates is open/closed/occupied by someone(Eg: Alien/Bot/Captain)
    :return: None
    """
    root = Tk()
    table = ttk.Frame(root)
    table.grid()
    ship_dimension = len(ship_layout)
    for row in range(ship_dimension):
        for col in range(ship_dimension):
            label = ttk.Label(table, text=ship_layout[row][col], borderwidth=1, relief="solid")
            label.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
    root.mainloop()


def run_simulation_for_n1_crew_members_n2_aliens(ship_dim: int, number_of_aliens: int, number_of_crew_members: int,
                                                 k: int, sampling_index_per_layout: int = 1,
                                                 sampling_index: int = 1, is_show_tkinter: bool = True):
    """
    :param k: radius for the bot's alien sensor
    :param number_of_aliens: number of aliens
    :param number_of_crew_members: number of crew members
    :param ship_dim:
    :param sampling_index_per_layout:
    :param sampling_index:
    :param is_show_tkinter:
    :return:
    """
    ship = Ship(ship_dim)
    ship_layout, root_open_square = ship.generate_ship_layout()
    spawner = Spawner(ship_layout, root_open_square)
    ship_layout, bot_init_coordinates = spawner.spawn_bot()
    ship_layout, alien_positions = spawner.spawn_aliens(number_of_aliens, bot_init_coordinates,k)
    ship_layout, crew_member_positions = spawner.spawn_crew_members(number_of_crew_members)
    show_tkinter(ship_layout)
    init_belief_matrix_for_one_crewmate = initialize_belief_matrix_for_one_crewmate(ship_layout)
    show_tkinter(init_belief_matrix_for_one_crewmate)
    init_belief_matrix_for_one_alien = initialize_belief_matrix_for_one_alien(ship_layout,bot_init_coordinates,k)
    show_tkinter(init_belief_matrix_for_one_alien)


def plot_metric(y, x, y_label, x_label, title):
    fig, (graph) = plt.subplots(1, 1, figsize=(12, 10))
    for bot in y:
        logging.info(f'The bot:{bot}')
        graph.plot(x, y[bot], label=bot)
    graph.set_title(title)
    graph.set_xlabel(x_label)
    graph.set_ylabel(y_label)
    graph.legend()
    plt.show()
