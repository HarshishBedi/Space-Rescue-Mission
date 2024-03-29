import os

import numpy as np

from src.Utilities.Simulation import run_simulation_for_n1_crew_members_n2_aliens, data_collection


def main():
    k_range = [1]
    alpha_range = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    data_collection(ship_dim=35, number_of_aliens=1, number_of_crew_members=2, k_range=k_range, alpha_range=alpha_range,
                    sampling_index=10, sampling_index_per_layout=10,
                    bot_types=['BOT1','BOT2'],
                    is_show_tkinter=False, is_gen_pdf=False)


if __name__ == '__main__':
    main()
