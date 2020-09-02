# Distributional Reinforcement Learning with Quantile Regression #

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
import os
import platform

import sim_environment
from logger import Logger
from rl_utils import ReplayMemory, huber

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 8, 3
rcParams['figure.dpi'] = 300

TMPATH = ".\pt_trainedmodel\ptmodel.pth"


class Network(nn.Module):
    def __init__(self, len_state, num_quant, num_actions):
        nn.Module.__init__(self)

        self.num_quant = num_quant
        self.num_actions = num_actions

        self.layer1 = nn.Linear(len_state, 256)
        self.layer2 = nn.Linear(256, num_actions * num_quant)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x.view(-1, self.num_actions, self.num_quant)

    def select_action(self, state, a_space, epsilon):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state])
        action = torch.randint(a_space[0], a_space[-1] + 1, (1,))
        if random.random() > epsilon:
            action = (self.forward(state).mean(2)[0, [a_space]]).max(1)[1] + a_space[0]
        return int(action)


# Environment and algorithm parameters

NUM_QUANTS = 5
STATE_LEN = 15
NUM_ACTIONS = 15
GAMMA = 0.6
EPS_START, EPS_END, EPS_DECAY = 0.9, 0.1, 1000
BATCH_SIZE = 32
REPLAY_MEM_SIZE = 500
NN_SYNC_FREQ = 500

# Instantiate Neural Networks
Z = Network(len_state=STATE_LEN, num_quant=NUM_QUANTS, num_actions=NUM_ACTIONS)
Ztgt = Network(len_state=STATE_LEN, num_quant=NUM_QUANTS, num_actions=NUM_ACTIONS)
optimizer = optim.Adam(Z.parameters(), 1e-3)


# Training
# Inputs - Nruns: No. of runs of training
# Outputs - None
def qr_dqn_train(Nruns):

    print("Running QR-DQN Training")

    # Quantiles
    tau = torch.Tensor((2 * np.arange(NUM_QUANTS) + 1) / (2.0 * NUM_QUANTS)).view(1, -1)

    logger = Logger('q-net', fmt={'loss': '.5f'})

    steps_done = 0
    running_reward = 0
    for run in range(Nruns):
        t = 0
        sum_reward = 0.0
        memory = ReplayMemory(REPLAY_MEM_SIZE)      # Initialize Replay buffer
        sim_environment.start_new_run(run)
        state = initial_state_generate()
        while True:
            intersection = t % 4
            if intersection == 3:
                a_space = [12, 13, 14]
            else:
                a_space = [4 * intersection, 4 * intersection + 1, 4 * intersection + 2,
                           4 * intersection + 3]

            action = Z.select_action(torch.Tensor([state]), a_space, calc_epsilon(steps_done))

            observ = sim_environment.take_action(action + 1)
            next_state = observ['next_state']
            reward = observ['rwd']
            done = 1 if reward == -100 else 0
            steps_done += 1
            t += 1

            if not done:
                memory.push(state, action, next_state, reward, float(done))
                sum_reward += reward

            if len(memory) < BATCH_SIZE:
                state = next_state
                continue

            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            theta = Z(states)[np.arange(BATCH_SIZE), actions]

            Znext = Ztgt(next_states).detach()
            Qnext_sa = Znext.mean(2)
            anext_max = torch.zeros([BATCH_SIZE], dtype=torch.long)
            for i in range(BATCH_SIZE):
                next_aspace = calc_next_aspace(int(actions[i]))
                temp = Qnext_sa[i, :]
                anext_max[i] = temp[next_aspace].max(0)[1] + next_aspace[0]

            Znext_max = Znext[np.arange(BATCH_SIZE), anext_max]
            Ttheta = rewards + GAMMA * (1 - dones) * Znext_max

            diff = Ttheta.t().unsqueeze(-1) - theta
            loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

            if steps_done % NN_SYNC_FREQ == 0:
                Ztgt.load_state_dict(Z.state_dict())

            if done:
                running_reward = sum_reward if not running_reward else 0.2*sum_reward + 0.8*running_reward
                logger.add(run + 1, steps=t, running_reward=running_reward, loss=loss.data.numpy())
                logger.iter_info()
                break

    torch.save(Z.state_dict(), TMPATH)

    return


# Live with value distribution image saving
# Inputs - load: 1 = Load and use a previously saved trained model, 0 = use current trained model
# Outputs - None
def qr_dqn_live(load):

    print("Running QR-DQN Live")

    # delete any existing images
    if platform.system() == 'Windows':
        os.system("del .\img\*.png")
    elif platform.system() == 'Linux':
        os.system("rm .\img\*.png")

    if load:
        model = Network(len_state=STATE_LEN, num_quant=NUM_QUANTS, num_actions=NUM_ACTIONS)
        model.load_state_dict(torch.load(TMPATH))
        model.eval()
    else:
        model = Network(len_state=STATE_LEN, num_quant=NUM_QUANTS, num_actions=NUM_ACTIONS)
        model.load_state_dict(Z.state_dict())
        model.eval()

    t = 0
    sim_environment.start_new_run(0)
    state = initial_state_generate()
    plt.show()
    plt.ion()
    while True:
        plt.clf()
        plt.title('step = %s' % t)

        intersection = t % 4
        if intersection == 3:
            a_space = [12, 13, 14]
        else:
            a_space = [4 * intersection, 4 * intersection + 1, 4 * intersection + 2,
                       4 * intersection + 3]

        action = model.select_action(torch.Tensor([state]), a_space, 0)

        observ = sim_environment.take_action(action + 1)
        state = observ['next_state']
        reward = observ['rwd']
        done = 1 if reward == -100 else 0
        t += 1

        Zval = model(torch.Tensor([state])).detach().numpy()
        for i in range(NUM_ACTIONS):
            x, y = get_plot(Zval[0][i])
            plt.plot(x, y, label='%s Q=%.1f' % (i + 1, Zval[0][i].mean()))
            plt.legend(bbox_to_anchor=(1.1, 1.1), ncol=NUM_ACTIONS, prop={'size': 3})

        if done: break

        plt.savefig('./img/%s.png' % t)
        plt.pause(0.001)

    plt.close()

    print("Steps = ", t)

    return


# Live with no image saving
# Inputs - Nruns: No. of live runs
#          load: 1 = Load and use a previously saved trained model, 0 = use current trained model
# Outputs - None
def qr_dqn_live_noplots(Nruns, load):

    print("Running QR-DQN Live (no plots)")

    if load:
        model = Network(len_state=STATE_LEN, num_quant=NUM_QUANTS, num_actions=NUM_ACTIONS)
        model.load_state_dict(torch.load(TMPATH))
        model.eval()
    else:
        model = Network(len_state=STATE_LEN, num_quant=NUM_QUANTS, num_actions=NUM_ACTIONS)
        model.load_state_dict(Z.state_dict())
        model.eval()

    for run in range(Nruns):
        t = 0
        sim_environment.start_new_run(run)
        state = initial_state_generate()
        while True:
            intersection = t % 4
            if intersection == 3:
                a_space = [12, 13, 14]
            else:
                a_space = [4 * intersection, 4 * intersection + 1, 4 * intersection + 2,
                           4 * intersection + 3]

            action = model.select_action(torch.Tensor([state]), a_space, 0)

            observ = sim_environment.take_action(action + 1)
            state = observ['next_state']
            reward = observ['rwd']
            done = 1 if reward == -100 else 0
            t += 1

            if done: break

        print("Steps = ", t)

    return


# Calculate epsilon for current step
def calc_epsilon(steps):

    epsilon = EPS_START / math.ceil((steps + 1) / EPS_DECAY)

    if epsilon < EPS_END:
        epsilon = EPS_END

    return epsilon


# Gives the next action space given the current action taken
def calc_next_aspace(curr_action):

    if curr_action < 4:
        next_aspace = [4, 5, 6, 7]
    elif curr_action < 8:
        next_aspace = [8, 9, 10, 11]
    elif curr_action < 12:
        next_aspace = [12, 13, 14]
    else:
        next_aspace = [0, 1, 2, 3]

    return next_aspace


# Generate a random initial state for the simulation
def initial_state_generate():
    for j in range(np.random.choice([4, 8, 12, 16, 20])):
        env_dict = sim_environment.take_action(0)
    return env_dict['next_state']


# Calculate action-value distribution curves
def get_plot(q):
    eps, p = 1e-8, 0
    x, y = [q[0] - np.abs(0.2 * q[0])], [0]

    for i in range(0, len(q)):
        x += [q[i] - eps, q[i]]
        y += [p, p + 1 / len(q)]
        p += 1 / len(q)

    x += [q[i] + eps, q[i] + np.abs(0.2 * q[i])]
    y += [1.0, 1.0]

    return x, y


# Print the NN weights & bias
def print_weights():

    model = Network(len_state=STATE_LEN, num_quant=NUM_QUANTS, num_actions=NUM_ACTIONS)
    model.load_state_dict(torch.load(TMPATH))
    model.eval()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])

    return
