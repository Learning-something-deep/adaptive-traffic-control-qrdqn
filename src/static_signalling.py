# Implements cyclic change of signals across intersections
# Takes input as no. of runs

import sim_environment
import numpy as np


# Cyclic signal change
def static_signalling(Nruns):

    print("Running Static signalling")

    for run in range(Nruns):
        print("Run " + str(run + 1))
        sim_environment.start_new_run(run)
        initial_state_generate()
        curr_a = 1      # cyclic test
        t = 0
        counter = [1, 0, 0, 0]
        while True:
            next_intersection = (t + 1) % 4
            if next_intersection == 3:
                a_space = [13, 14, 15]
            else:
                a_space = [4 * next_intersection + 1, 4 * next_intersection + 2, 4 * next_intersection + 3,
                           4 * next_intersection + 4]
            next_a = a_space[counter[next_intersection]]

            if counter[next_intersection] != len(a_space) - 1:
                counter[next_intersection] += 1
            else:
                counter[next_intersection] = 0

            env_param = sim_environment.take_action(curr_a)
            r = env_param['rwd']
            if r == -100:
                print("End of simulation at t = " + str(t))
                break
            curr_a = next_a
            t += 1
    return


# Generate a random initial state for the simulation
def initial_state_generate():
    for j in range(np.random.choice([4, 8, 12, 16, 20])):
        env_dict = sim_environment.take_action(0)
    return env_dict['next_state']
