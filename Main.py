import argparse
from src.Utilities.Simulation import run_simulation_for_n1_crew_members_n2_aliens, data_collection


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='1.0')
    subparsers = parser.add_subparsers(dest='command', required=True)
    bot1Parse = subparsers.add_parser('run_simulation',
                                      help='Run simulation with Bot1')
    # edit the array to include the values of k you want to collect the data for
    k_range = [1]
    # edit the alpha_range array to include the values you want to run the simulation over
    alpha_range = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    args = parser.parse_args()
    # edit the bots to include the bots you want to simulate and compare
    bots = ['BOT6', 'BOT7', 'BOT8']
    # edit the number_of_aliens, number_of_crew_members values for different simulation
    if args.command == 'run_simulation':
        data_collection(ship_dim=35, number_of_aliens=2, number_of_crew_members=2, k_range=k_range,
                        alpha_range=alpha_range,
                        sampling_index=100, sampling_index_per_layout=5,
                        bot_types=bots,
                        is_show_tkinter=False, is_gen_pdf=False)


if __name__ == '__main__':
    main()
