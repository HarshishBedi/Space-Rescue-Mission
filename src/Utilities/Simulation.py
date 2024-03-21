import logging
from tkinter import Tk, ttk
import matplotlib.pyplot as plt

from src.BeliefUpdates.Aliens.OneAlien import initialize_belief_matrix_for_one_alien
from src.BeliefUpdates.CrewMembers.OneCrewMember import initialize_belief_matrix_for_one_crew_member
from src.Bots.Bot1 import Bot1
from src.Bots.Bot3 import Bot3
from src.Utilities.Alien import alien_step
from src.Utilities.Alien_Sensor import alien_sensor
from src.Utilities.Crew_Member_Sensor import Crew_Member_Sensor
from src.Utilities.Ship import Ship
from src.Utilities.Spawner import Spawner
from src.Utilities.Status import Status
from src.Utilities.utility import export_to_pdf


def show_tkinter(ship_layout: list[list[str]]):
    """
    :param belief:
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
                                                 k: int, alpha: float, sampling_index_per_layout: int = 1,
                                                 sampling_index: int = 1, is_show_tkinter: bool = True):
    """
    :param alpha:
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
    for _ in range(sampling_index):
        ship_layout, root_open_square = ship.generate_ship_layout()
        for _ in range(sampling_index_per_layout):
            spawner = Spawner(ship_layout, root_open_square)
            ship_layout, bot_init_coordinates = spawner.spawn_bot()
            ship_layout, alien_positions = spawner.spawn_aliens(number_of_aliens, bot_init_coordinates, k)
            ship_layout, crew_member_positions = spawner.spawn_crew_members(number_of_crew_members)
            init_belief_matrix_for_one_crewmate = initialize_belief_matrix_for_one_crew_member(ship_layout)
            init_belief_matrix_for_one_alien = initialize_belief_matrix_for_one_alien(ship_layout, bot_init_coordinates,
                                                                                      k)
            bot1 = Bot1(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate,
                        alpha, k)
            crew_member_sensor = Crew_Member_Sensor(crew_member_positions, alpha)
            status = Status.INPROCESS
            number_of_steps = 0
            while status == Status.INPROCESS:
                alien_sensed = alien_sensor(bot1.position, alien_positions, k, ship_dim=ship_dim)
                crew_member_beep = crew_member_sensor.crew_members_beep(bot1.position)
                print(f'Crew member beep received:{crew_member_beep}')
                print(f'Alien sensed:{alien_sensed}')
                export_to_pdf(ship_layout)
                if is_show_tkinter:
                    show_tkinter(ship_layout)
                    show_tkinter(bot1.crew_member_belief)
                cb, ab = bot1.update_beliefs(ship_layout, alien_sensed, crew_member_beep)
                status, ship_layout, _ = bot1.bot_step(ship_layout)
                if status != Status.INPROCESS:
                    break
                status, ship_layout, alien_positions = alien_step(ship_layout, alien_positions)
                number_of_steps += 1
            if status == Status.SUCCESS:
                print(f'Bot succeeded after {number_of_steps} steps')
            elif status == Status.FAILURE:
                print(f'Bot failed after {number_of_steps} steps')


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
