import numpy as np

from src.Utilities.Simulation import run_simulation_for_n1_crew_members_n2_aliens


def main():
    run_simulation_for_n1_crew_members_n2_aliens(35, 2, 2, 1,1,
                                                 bot_type='BOT7',
                                                 is_show_tkinter=False)


if __name__ == '__main__':
    main()