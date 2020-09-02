# Implements the Longest Queue First (LQF) algorithm on the traffic network intersections
# Takes as input No. of runs

# Depends on: sim_environment.py

import sim_environment
import random
import numpy as np


# Desc: Runs Longest Queue First(LQF) algorithm on each intersection of the traffic network.
# Inputs - Nruns: No. of runs
# Outputs - None
def lqf(Nruns):

    print("Running LQF")

    for run in range(Nruns):
        print("Run " + str(run + 1))
        sim_environment.start_new_run(run)
        initial_state_generate()
        curr_a = random.randint(1, 4)
        t = 0
        while True:
            next_intersection = (t + 1) % 4
            env_param = sim_environment.take_action(curr_a)
            next_s = env_param['next_state']
            r = env_param['rwd']
            if r == -100:
                print("End of simulation at t = " + str(t))
                break

            if next_intersection == 3:
                a_space = [13, 14, 15]
            else:
                a_space = [4 * next_intersection + 1, 4 * next_intersection + 2, 4 * next_intersection + 3,
                           4 * next_intersection + 4]

            q_next = []
            for a_temp in a_space:
                q_next.append(next_s[a_temp - 1])
            q_max = max(q_next)
            q_max_index = [i for i, j in enumerate(q_next) if j == q_max]
            rand_greedy_q = np.random.choice(q_max_index)
            next_a = a_space[rand_greedy_q]
            curr_a = next_a
            t += 1
    return


# Generate a random initial state for the simulation
def initial_state_generate():
    for j in range(np.random.choice([4, 8, 12, 16, 20])):
        env_dict = sim_environment.take_action(0)
    return env_dict['rwd'], env_dict['next_state']
