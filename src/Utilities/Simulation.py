import copy
import logging
import os
import shutil
import time
from tkinter import Tk, ttk
import matplotlib.pyplot as plt
import numpy as np

from src.BeliefUpdates.Aliens.OneAlien import initialize_belief_matrix_for_one_alien
from src.BeliefUpdates.Aliens.TwoAliens import initialize_belief_matrix_for_two_aliens
from src.BeliefUpdates.CrewMembers.OneCrewMember import initialize_belief_matrix_for_one_crew_member
from src.BeliefUpdates.CrewMembers.TwoCrewMembers import initialize_belief_matrix_for_two_crew_members
from src.Bots.Bot1 import Bot1
from src.Bots.Bot2 import Bot2
from src.Bots.Bot3 import Bot3
from src.Bots.Bot4 import Bot4
from src.Bots.Bot7 import Bot7
from src.Utilities.Alien import alien_step
from src.Utilities.Alien_Sensor import alien_sensor
from src.Utilities.CreateTableClass import PDF
from src.Utilities.Crew_Member_Sensor import Crew_Member_Sensor
from src.Utilities.Ship import Ship
from src.Utilities.Spawner import Spawner
from src.Utilities.Status import Status
from src.Utilities.utility import append_to_pdf, get_num_of_open_cells_in_ship, open_neighbor_cells_matrix


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
                                                 sampling_index: int = 1, bot_types: list[str] = [],
                                                 is_show_tkinter: bool = True, is_gen_pdf: bool = False):
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
    avg_num_steps_save_crew = 0
    num_of_success = 0
    avg_num_crew_saved = 0
    for i in range(sampling_index):
        ship_layout, root_open_square = ship.generate_ship_layout()
        open_neighbor_cells = open_neighbor_cells_matrix(ship_layout)
        print(f'Ship layout generated:{len(ship_layout)} and {len(ship_layout[0])}')
        spawner = Spawner(ship_layout, root_open_square)
        ship_layout, bot_init_coordinates = spawner.spawn_bot()
        ship_layout, alien_positions = spawner.spawn_aliens(number_of_aliens, bot_init_coordinates, k)
        ship_layout, crew_member_positions = spawner.spawn_crew_members(number_of_crew_members)
        for j in range(sampling_index_per_layout):
            for bot_type in bot_types:
                start_time = time.time()
                print('***************************')
                print(f'Running simulation for {i}th ship and {j}th time')

                temp_ship_layout = copy.deepcopy(ship_layout)
                temp_alien_positions = copy.deepcopy(alien_positions)
                num_crew_saved, number_of_steps, status = simulate_for_bot_k_alpha(temp_alien_positions, alpha,
                                                                                   bot_init_coordinates,
                                                                                   bot_type,
                                                                                   crew_member_positions,
                                                                                   is_gen_pdf, k,
                                                                                   number_of_crew_members,
                                                                                   open_neighbor_cells,
                                                                                   ship_dim, temp_ship_layout)
                avg_num_crew_saved += num_crew_saved
                if status == Status.SUCCESS:
                    print(f'Bot succeeded after {number_of_steps} steps')
                    avg_num_steps_save_crew += number_of_steps
                    num_of_success += 1
                elif status == Status.FAILURE:
                    print(f'Bot failed after {number_of_steps} steps')
                print(f'Completed simulation for {i}th ship and {j}th time in {time.time()-start_time} seconds')
                print('***************************')
    success_prob = num_of_success / (sampling_index * sampling_index_per_layout)
    avg_num_steps_save_crew = avg_num_steps_save_crew / num_of_success if num_of_success != 0.0 else float('inf')
    avg_num_crew_saved = avg_num_crew_saved / (sampling_index * sampling_index_per_layout)
    return avg_num_steps_save_crew, success_prob, avg_num_crew_saved


def simulate_for_bot_k_alpha(alien_positions, alpha, bot_init_coordinates, bot_type, crew_member_positions, is_gen_pdf,
                             k, number_of_crew_members, open_neighbor_cells, ship_dim, ship_layout):
    num_crew_saved = 0
    (init_belief_matrix_for_one_crewmate,
     init_belief_matrix_for_one_alien) = init_belief(bot_type=bot_type, ship_layout=ship_layout,
                                                     bot_init_coordinates=bot_init_coordinates, k=k)
    bot = get_bot_object(bot_init_coordinates, init_belief_matrix_for_one_alien,
                         init_belief_matrix_for_one_crewmate, alpha, k, number_of_crew_members, bot_type,
                         open_neighbor_cells)
    crew_member_sensor = Crew_Member_Sensor(crew_member_positions, alpha)
    status = Status.INPROCESS
    number_of_steps = 0
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Times", size=10)
    if is_gen_pdf:
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
    while status == Status.INPROCESS:
        start_time = time.time()
        alien_sensed = alien_sensor(bot.position, alien_positions, k, ship_dim=ship_dim)
        crew_member_beep = crew_member_sensor.crew_members_beep(bot.position)
        # print(f'Crew member beep received:{crew_member_beep}')
        # print(f'Alien sensed:{alien_sensed}')
        bot.update_beliefs(ship_layout, alien_sensed, crew_member_beep)
        if is_gen_pdf:
            pdf.add_page()
            pdf.set_font("Times", size=10)
            pdf.multi_cell(100, pdf.font_size * 2.5, 'Crew Member beep:' + str(crew_member_beep),
                           border=0, align='j', ln=3, max_line_height=pdf.font_size)
            pdf.ln(pdf.font_size * 2.5)
            pdf.multi_cell(100, pdf.font_size * 2.5, 'Alien Sensed:' + str(alien_sensed), border=0,
                           align='j',
                           ln=3, max_line_height=pdf.font_size)
            pdf.ln(pdf.font_size * 2.5)
        status, ship_layout, _, num_crew_saved = bot.bot_step(ship_layout)
        if status != Status.INPROCESS:
            break
        status, ship_layout, alien_positions = alien_step(ship_layout, alien_positions)
        if is_gen_pdf:
            pdf.multi_cell(100, pdf.font_size * 2.5,
                           'Goal Postion:' + str(bot.goal[0]) + ',' + str(bot.goal[1]),
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
        # print(f'Timestep {number_of_steps} completed in {time.time() - start_time} seconds')
    pdf.output('table_class.pdf')
    return num_crew_saved, number_of_steps, status


def data_collection(ship_dim: int, number_of_aliens: int, number_of_crew_members: int,
                    k_range: list[int], alpha_range: list[float], bot_types: list[str],
                    sampling_index_per_layout: int = 1,
                    sampling_index: int = 1,
                    is_show_tkinter: bool = True, is_gen_pdf: bool = False):
    if bot_types is None:
        bot_types = []
    avg_num_steps_save_crew = np.zeros((len(k_range),len(alpha_range)), float)
    success_prob = np.zeros((len(k_range),len(alpha_range)), float)
    avg_num_crew_saved = np.zeros((len(k_range),len(alpha_range)), float)
    for i in range(len(k_range)):
        for j in range(len(alpha_range)):
            alpha = alpha_range[j]
            k = k_range[i]
            (avg_num_steps_save_crew[i][j],
             success_prob[i][j],
             avg_num_crew_saved[i][j]) = run_simulation_for_n1_crew_members_n2_aliens(ship_dim, number_of_aliens,
                                                                                      number_of_crew_members, k, alpha,
                                                                                      sampling_index_per_layout,
                                                                                      sampling_index, bot_types,
                                                                                      is_show_tkinter, is_gen_pdf)
    save_metric_plots(alpha_range, avg_num_crew_saved, avg_num_steps_save_crew, k_range, success_prob)


def save_metric_plots(alpha_range, avg_num_crew_saved, avg_num_steps_save_crew, k_range, success_prob):
    results_root_dir = ('C://Users//ASUS//OneDrive//Desktop//MS Rutgers//Spring 2024//Intro to '
                        'AI//Projects//Space-Rescue-Mission//Results')

    if os.path.exists(results_root_dir):
        shutil.rmtree(results_root_dir)
    os.mkdir(results_root_dir)
    for i in range(len(k_range)):
        k_dir = results_root_dir + '//alpha_range_k' + str(k_range[i])
        os.mkdir(k_dir)
        avg_num_steps_save_crew_path = k_dir + '//metric-1.pdf'
        plot_metric(avg_num_steps_save_crew[i, :], alpha_range, 'Average Number of steps to save crew',
                    'Alpha values',
                    'Average number of moves needed to rescue all crew members over range of alpha values '
                    'and k=' + str(k_range[i]), avg_num_steps_save_crew_path)
        success_prob_path = k_dir + '//metric-2.pdf'
        plot_metric(success_prob[i, :], alpha_range, 'Success Probability of the bot',
                    'Alpha values',
                    'Probability of successfully avoiding the alien and rescuing the crew over range of alpha values '
                    'and k=' + str(k_range[i]), success_prob_path)
        avg_num_crew_saved_path = k_dir + '//metric-3.pdf'
        plot_metric(avg_num_crew_saved[i, :], alpha_range, 'Average number of crew members saved',
                    'Alpha values',
                    'Average number of crew members saved over range of alpha values and k=' + str(k_range[i]),
                    avg_num_crew_saved_path)
    for i in range(len(alpha_range)):
        alpha_dir = results_root_dir + '//k_range_alpha' + str(alpha_range[i])
        os.mkdir(alpha_dir)
        avg_num_steps_save_crew_path = alpha_dir + '//metric-1.pdf'
        plot_metric(avg_num_steps_save_crew[:, i], k_range, 'Average Number of steps to save crew',
                    'Alpha values',
                    'Average number of moves needed to rescue all crew members over range of k values '
                    'and alpha =' + str(alpha_range[i]), avg_num_steps_save_crew_path)
        success_prob_path = alpha_dir + '//metric-2.pdf'
        plot_metric(success_prob[:, i], k_range, 'Success Probability of the bot',
                    'Alpha values',
                    'Probability of successfully avoiding the alien and rescuing the crew over range of k values '
                    'and alpha =' + str(alpha_range[i]), success_prob_path)
        avg_num_crew_saved_path = alpha_dir + '//metric-3.pdf'
        plot_metric(avg_num_crew_saved[:, i], k_range, 'Average number of crew members saved',
                    'Alpha values',
                    'Average number of crew members saved over range of k values and alpha =' + str(alpha_range[i]),
                    avg_num_crew_saved_path)


def plot_metric(y, x, y_label, x_label, title, file_path):
    fig, (graph) = plt.subplots(1, 1, figsize=(12, 10))
    print(title)
    print(y)
    print(x)
    for i in range(1):
        logging.info(f'The bot:{i}')
        graph.plot(x, y, label='1')
    graph.set_title(title)
    graph.set_xlabel(x_label)
    graph.set_ylabel(y_label)
    graph.legend()
    plt.savefig(file_path)
    plt.close()


def convert_list_float_to_str(belief: list[list[float]]):
    str_belief = []
    for row in belief:
        str_belief.append([])
        for val in row:
            str_belief[-1].append(str(val))
    return str_belief


def get_bot_object(bot_init_coordinates: tuple[int, int], init_belief_matrix_for_one_alien,
                   init_belief_matrix_for_one_crewmate,
                   alpha: float, k: int, number_of_crew_members: int, bot_type: str, open_neighbor_cells):
    if bot_type == 'BOT1':
        return Bot1(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate,
                    alpha, k)
    elif bot_type == 'BOT2':
        return Bot2(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate,
                    alpha, k)
    elif bot_type == 'BOT3':
        return Bot3(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate, alpha,
                    k, number_of_crew_members)
    elif bot_type == 'BOT7':
        return Bot7(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate, alpha,
                    k, number_of_crew_members, open_neighbor_cells=open_neighbor_cells)
    elif bot_type == 'BOT4':
        return Bot4(bot_init_coordinates, init_belief_matrix_for_one_alien, init_belief_matrix_for_one_crewmate, alpha,
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
