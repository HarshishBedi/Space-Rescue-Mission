# Space-Rescue-Mission
We are designing different bots which help the space roomba whose crew mates are in danger because of the threat of aliens. We compare the performance of different bots in different scenarios, with different approaches
For running the simulation, install the requirements.
Then open terminal and navigate to the folder containing the Main.py file 
run the command: python Main.py run_simulation


For changing the simulation to run the simulation for different alpha values, or k values or bots:
edit the Main.py file:
edit the k_range array to include the values of k you want to collect the data for
edit the alpha_range array to include the values you want to run the simulation over
edit the bots to include the bots you want to simulate and compare 
(for example for running a the simulation 1 for one alien and 1 crew member: edit bots array to ['BOT1','BOT2'] and number_of_crew_members=1 and number_of_aliens=1)

for improving the speed of execution:
  you can edit the number of processes pooled in the Utilities/Simulation.py file, line 78: ProcessPoolExecutor(10). If the RAM permits, you can increase the number of processes upto 60.

