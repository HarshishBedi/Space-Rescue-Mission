import argparse
import os

import numpy as np

from src.Utilities.Simulation import run_simulation_for_n1_crew_members_n2_aliens, data_collection


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='1.0')
    subparsers = parser.add_subparsers(dest='command', required=True)
    bot1Parse = subparsers.add_parser('bot8',
                                      help='Run simulation with Bot1')
    k_range = [1,2,3,4,5,6,7,8,9,10]
    alpha_range = [0.25]
    # args = parser.parse_args()
    # if args.command == 'bot8':
    data_collection(ship_dim=35, number_of_aliens=1, number_of_crew_members=1, k_range=k_range,
                    alpha_range=alpha_range,
                    sampling_index=10, sampling_index_per_layout=10,
                    bot_types=['BOT6','BOT7','BOT8'],
                    is_show_tkinter=False, is_gen_pdf=False)
def k_largest_index_argpartition_v1(a, k):
    idx = np.argpartition(-a.ravel(), k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))


if __name__ == '__main__':
    main()