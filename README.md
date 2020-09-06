# Adaptive traffic control system using Deep Reinforcement Learning

This project implements **Quantile Regression Deep Q-Network (QR-DQN)** algorithm to tackle the traffic congestion problem. The python package dependencies for this project are listed under the requirements.txt file. Make sure that the required dependencies are installed.

We use SUMO [Simulation of Urban Mobility] V1.5 software to simulate traffic patterns on which the algorithm has been tested. Details of SUMO installation can be found at https://sumo.dlr.de/docs/Installing.html

We have used PyCharm and Anaconda for development of this project. Any IDE can be used, provided the dependencies are properly installed in your respective environments. The OS can be Windows/Linux.

Note: We implemented SARSA algorithm for traffic problem for the Reinforcement Learning course project. We use the obtained results for comparison with QR-DQN.


The project folder contains two directories: src and scripts
* The src directory contains all the source codes needed to run the project
* The scripts directory contains SUMO files describing the road network and routes. Also output files from SUMO will be generated in this directory


## Source tree description
The src folder contains the following files and sub-directories:
1. main.py: This is the main file of the project. It includes calls to static_signalling(), lqf(), qr_dqn_train(), qr_dqn_live_noplots() and plot_all_metrics() functions
2. lqf.py: This file implements the Longest Queue First algorithm.
3. static_signalling.py: This file implements the Static Signalling [SS] algorithm.
4. qr_dqn.py: This script contains the following modules:
	1. qr_dqn_train(): This module implements the QR-DQN algorithm. 
	2. qr_dqn_live_noplots(): This module makes use of the trained Neural Network from qr_dqn_train() module to test the performance of the QR-DQN algorithm.
5. sim_environment.py: This file acts as the RL environment. It returns the next state and reward given an action.
6. plot_metrics.py: This file contains modules that plot the following metrics: Queue Length, Waiting Time, Time Loss and Dispersion Time. It also saves the metrics into the "datapoints" directory present in the src folder.
7. generate_plots.py: This file plots the performance comparison graphs of QR-DQN, LQF, SS and SARSA algorithms using various metrics. 
8. rl_utils.py, logger.py: Contains various helper functions.
9. datapoints: This folder contains the saved datapoints of the averaged metrics. Ensure that this directory is present inside the src folder.
10. pt_trainedmodel: This folder contains the saved Neural Network weights after training. Ensure that this directory is present inside the src folder.

## Procedure to run the project
1. Set the Nruns parameter for each algo in the main.py file to specify the no. of training/trial runs. Recommended minimum Nruns=25.
2. Run main.py. This will run SS Algo for "Nruns" trials and LQF Algo for "Nruns" trials. Then training for QR-DQN will start. Based on the trained weights, QR-DQN Live will perform "Nruns" trials. All the performance metrics will be stored into the "/datapoints" sub-directory at the end of "Nruns" trials.
3. Run generate_plots.py to get the performance comparison graphs.
