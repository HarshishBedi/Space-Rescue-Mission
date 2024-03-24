import logging
from tkinter import Tk, ttk
import matplotlib.pyplot as plt

from src.BeliefUpdates.Aliens.OneAlien import initialize_belief_matrix_for_one_alien
from src.BeliefUpdates.Aliens.TwoAliens import initialize_belief_matrix_for_two_aliens
from src.BeliefUpdates.CrewMembers.OneCrewMember import initialize_belief_matrix_for_one_crew_member
from src.BeliefUpdates.CrewMembers.TwoCrewMembers import initialize_belief_matrix_for_two_crew_members
from src.Bots.Bot1 import Bot1
from src.Bots.Bot3 import Bot3
from src.Utilities.Alien import alien_step
from src.Utilities.Alien_Sensor import alien_sensor
from src.Utilities.CreateTableClass import PDF
from src.Utilities.Crew_Member_Sensor import Crew_Member_Sensor
from src.Utilities.Ship import Ship
from src.Utilities.Spawner import Spawner
from src.Utilities.Status import Status
from src.Utilities.utility import append_to_pdf, get_num_of_open_cells_in_ship


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
                                                 sampling_index: int = 1, bot_type: str = 'BOT1',
                                                 is_show_tkinter: bool = True):
    """
    :param bot_type:
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
        print(f'Ship layout generated:{len(ship_layout)} and {len(ship_layout[0])}')
        for _ in range(sampling_index_per_layout):
            spawner = Spawner(ship_layout, root_open_square)
            ship_layout, bot_init_coordinates = spawner.spawn_bot()
            ship_layout, alien_positions = spawner.spawn_aliens(number_of_aliens, bot_init_coordinates, k)
            ship_layout, crew_member_positions = spawner.spawn_crew_members(number_of_crew_members)
            init_belief_matrix_for_one_crewmate = initialize_belief_matrix_for_one_crew_member(ship_layout)
            (init_belief_matrix_for_one_crewmate,
             init_belief_matrix_for_one_alien) = init_belief(bot_type=bot_type, ship_layout=ship_layout,
                                                             bot_init_coordinates=bot_init_coordinates, k=k)
            bot = get_bot_object(bot_init_coordinates, init_belief_matrix_for_one_alien,
                                 init_belief_matrix_for_one_crewmate, alpha, k, number_of_crew_members, bot_type)
            crew_member_sensor = Crew_Member_Sensor(crew_member_positions, alpha)
            status = Status.INPROCESS
            number_of_steps = 0
            pdf = PDF()
            pdf.add_page()
            pdf.set_font("Times", size=10)

            pdf.multi_cell(100, pdf.font_size * 2.5,
                           'Number of open cells:' + str(get_num_of_open_cells_in_ship(ship_layout)),
                           border=0, align='j', ln=3,
                           max_line_height=pdf.font_size)
            pdf.ln(pdf.font_size * 2.5)
            pdf.create_table(table_data=ship_layout, title='Init Ship Layout' + str(number_of_steps),
                             cell_width='even')
            pdf.create_table(table_data=convert_list_float_to_str(bot.crew_member_belief),
                             title='Init Crew Member Belief ' +
                                   str(number_of_steps),
                             cell_width='even')
            pdf.create_table(table_data=convert_list_float_to_str(bot.alien_belief),
                             title='Init Alien belief ' + str(number_of_steps),
                             cell_width='even')
            print(ship_layout)
            while status == Status.INPROCESS:
                alien_sensed = alien_sensor(bot.position, alien_positions, k, ship_dim=ship_dim)
                crew_member_beep = crew_member_sensor.crew_members_beep(bot.position)
                print(f'Crew member beep received:{crew_member_beep}')
                print(f'Alien sensed:{alien_sensed}')
                cb, ab = bot.update_beliefs(ship_layout, alien_sensed, crew_member_beep)
                pdf.add_page()
                pdf.set_font("Times", size=10)
                pdf.multi_cell(100, pdf.font_size * 2.5, 'Crew Member beep:' + str(crew_member_beep),
                               border=0, align='j', ln=3, max_line_height=pdf.font_size)
                pdf.ln(pdf.font_size * 2.5)
                pdf.multi_cell(100, pdf.font_size * 2.5, 'Alien Sensed:' + str(alien_sensed), border=0, align='j',
                               ln=3, max_line_height=pdf.font_size)
                pdf.ln(pdf.font_size * 2.5)
                status, ship_layout, _ = bot.bot_step(ship_layout)
                if status != Status.INPROCESS:
                    break
                status, ship_layout, alien_positions = alien_step(ship_layout, alien_positions)
                pdf.multi_cell(100, pdf.font_size * 2.5, 'Goal Postion:' + str(bot.goal[0]) + ',' + str(bot.goal[1]),
                               border=0, align='j', ln=3,
                               max_line_height=pdf.font_size)
                pdf.ln(pdf.font_size * 2.5)
                pdf.multi_cell(100, pdf.font_size * 2.5, 'Path to goal:' + str(bot.path),
                               border=0, align='j', ln=3,
                               max_line_height=pdf.font_size)
                pdf.ln(pdf.font_size * 2.5)
                pdf.create_table(table_data=ship_layout, title='Ship Layout at time ' + str(number_of_steps),
                                 cell_width='even')
                pdf.create_table(table_data=convert_list_float_to_str(bot.crew_member_belief),
                                 title='Crew Member Belief  at time ' +
                                       str(number_of_steps),
                                 cell_width='even')
                pdf.create_table(table_data=convert_list_float_to_str(bot.alien_belief),
                                 title='Alien belief  at time ' + str(number_of_steps),
                                 cell_width='even')
                number_of_steps += 1
            pdf.output('table_class.pdf')
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


def convert_list_float_to_str(belief: list[list[float]]):
    str_belief = []
    for row in belief:
        str_belief.append([])
        for val in row:
            str_belief[-1].append(str(val))
    return str_belief


def get_bot_object(bot_init_coordinates: tuple[int, int], init_belief_matrix_for_one_alien,
                   init_belief_matrix_for_one_crewmate,
                   alpha: float, k: int, number_of_crew_members: int, bot_type: str):
    if bot_type == 'BOT1':
        return Bot1(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate,
                    alpha, k)
    elif bot_type == 'BOT2':
        return None
    elif bot_type == 'BOT3':
        return Bot3(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate, alpha,
                    k, number_of_crew_members)


def init_belief(bot_type: str, ship_layout, bot_init_coordinates, k):
    if bot_type == 'BOT1' or bot_type == 'BOT2' or bot_type == 'BOT3' or bot_type == 'BOT6':
        return (initialize_belief_matrix_for_one_crew_member(ship_layout),
                initialize_belief_matrix_for_one_alien(ship_layout, bot_init_coordinates, k))
    elif bot_type == 'BOT4' or bot_type == 'BOT5':
        return (initialize_belief_matrix_for_two_crew_members(ship_layout),
                initialize_belief_matrix_for_one_alien(ship_layout, bot_init_coordinates, k))
    elif bot_type == 'BOT7' or bot_type == 'BOT8':
        return (initialize_belief_matrix_for_two_crew_members(ship_layout),
                initialize_belief_matrix_for_two_aliens(ship_layout, bot_init_coordinates, k))
