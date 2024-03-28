import os

import numpy as np

from src.Utilities.Simulation import run_simulation_for_n1_crew_members_n2_aliens, data_collection


def main():
    k_range = [1]
    alpha_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.91]
    data_collection(ship_dim=35, number_of_aliens=1, number_of_crew_members=1, k_range=k_range, alpha_range=alpha_range,
                    sampling_index=1, sampling_index_per_layout=1,
                    bot_types=['BOT2'],
                    is_show_tkinter=False)


if __name__ == '__main__':
    main()
