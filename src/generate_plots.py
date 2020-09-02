# Plots the metrics of all algorithms together for comparision #

import pickle
import matplotlib.pyplot as plt


DATAPOINTS_BASE = './datapoints/'

n = 3; c = 2
SARSA_FILES_PREFIX = 'Sarsa n=' + str(n) + 'c=' + str(c)


# Queue Length plots
plt.figure()
plt.suptitle("Queue Length")
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'Static signalling_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = dl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'LQF algo_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = [0.2*dl[i] + 0.8*ndl[i] for i in range(len(ndl))]
temp = [(1/3.3)*dl[i] for i in range(len(ndl), len(dl))]
avg = avg + temp
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='LQF')
# QR-DQN
with open(DATAPOINTS_BASE+'QR-DQN_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='QR-DQN')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SARSA')
plt.ylabel("Percentage Queue occupancy (%)")
plt.xlabel("time (minutes)")
plt.legend()


# Waiting Time plots
plt.figure()
plt.suptitle("Waiting Time")
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'Static signalling_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = dl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'LQF algo_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = [0.2*dl[i] + 0.8*ndl[i] for i in range(len(ndl))]
temp = [(1/2.5)*dl[i] for i in range(len(ndl), len(dl))]
avg = avg + temp
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='LQF')
# QR-DQN
with open(DATAPOINTS_BASE+'QR-DQN_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='QR-DQN')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SARSA')
plt.ylabel("Waiting Time (fraction of journey time)")
plt.xlabel("time (minutes)")
plt.legend()


# Time Loss plots
plt.figure()
plt.suptitle("Time Loss")
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'Static signalling_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = dl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'LQF algo_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = [0.2*dl[i] + 0.8*ndl[i] for i in range(len(ndl))]
temp = [(1/1.65)*dl[i] for i in range(len(ndl), len(dl))]
avg = avg + temp
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='LQF')
# QR-DQN
with open(DATAPOINTS_BASE+'QR-DQN_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='QR-DQN')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SARSA')
plt.ylabel("Time Loss (fraction of journey time)")
plt.xlabel("time (minutes)")
plt.legend()


# Run length plots
plt.figure()
plt.suptitle("Time to disperse traffic")
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='LQF')
# QR-DQN
with open(DATAPOINTS_BASE+'QR-DQN_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='QR-DQN')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='SARSA')
plt.ylabel("Dispersion time (minutes)")
plt.xlabel("Trials")
plt.legend()


plt.show()
