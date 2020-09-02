# Main module

# Depends on: qr_dqn.py, lqf_algo.py, static_signalling.py, plot_metrics.py

import sim_environment
import static_signalling
import lqf_algo
import qr_dqn
import plot_metrics
from matplotlib import pyplot as plt


# Static signalling
sim_environment.endis_sumo_guimode(1)
Nruns = 25
static_signalling.static_signalling(Nruns)
plot_metrics.plot_all_metrics("Static signalling")


# Longest Queue First (LQF) algorithm
sim_environment.endis_sumo_guimode(1)
Nruns = 25
lqf_algo.lqf(Nruns)
plot_metrics.plot_all_metrics("LQF algo")


# Train the QR-DQN model
sim_environment.endis_sumo_guimode(0)
Nruns = 25
qr_dqn.qr_dqn_train(Nruns)

# Try out performance
sim_environment.endis_sumo_guimode(1)
Nruns = 25; use_saved_model = 0
qr_dqn.qr_dqn_live_noplots(Nruns, use_saved_model)
plot_metrics.plot_all_metrics("QR-DQN")


plt.show()      # to prevent figures from closing
