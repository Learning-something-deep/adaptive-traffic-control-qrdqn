# Parses the output of SUMO simulation, and plots congestion-related performance metrics.
# Plots Average queue length at intersections, Average waiting time, Average time loss and Dispersion times (averaged
# across simulation runs). Uses the output dump file of SUMO for each run as input.

# Depends on: None

from lxml import etree as ET
import matplotlib.pyplot as plt
import pickle

TRIPINFO_FILE = "../scripts/txmap-tripinfo.xml"
QLENGTH_FILE = "../scripts/qlengths.xml"
DATAPOINTS_BASE = './datapoints/'

DEADLOCK_THRESH = 1300

STEP_SIZE = 0
run = 0

# 2D arrays to store metrics of all runs
timeLoss_all_runs = []
waitingTime_all_runs = []
qlength_all_runs = []
huber_loss_arr = []
running_rwrd_arr = []
running_avg_rwrd_arr = []


# Desc: Clears any existing data for all runs, and starts fresh.
# Inputs - step_size: The number of simulation steps that equal one time step
# Outputs - None
def init(step_size):

    global STEP_SIZE, run, timeLoss_all_runs, waitingTime_all_runs, qlength_all_runs

    STEP_SIZE = step_size
    run = 0

    timeLoss_all_runs = []
    waitingTime_all_runs = []
    qlength_all_runs = []

    return


# Desc: Parses the simulation output file at the end of run. Performs averaging, and stores various vehicle metrics
#       for the run.
# Inputs - skip_time: simulation steps spent in initial randomization phase
# Outputs - None
def record_metrics_of_run(skip_time):

    global run, timeLoss_all_runs, waitingTime_all_runs, qlength_all_runs

    # parse output and put the lines in an array
    elementList = []
    tree = ET.parse(TRIPINFO_FILE)
    root = tree.getroot()
    nlines = 0
    for element in root.iter('tripinfo'):
        elementList.append(element)
        nlines += 1

    # calculate the average metrics for each time step
    i = 0
    tstep = 0
    lineno = 0
    t = skip_time
    timeLoss_arr = []
    waitingTime_arr = []
    timeLoss_arr_temp = []
    waitingTime_arr_temp = []
    while lineno < nlines:

        element = elementList[lineno]
        arrival = float(element.get('arrival'))

        if arrival < skip_time:             # discard line
            lineno += 1
            continue
        elif arrival <= t + STEP_SIZE:      # read metrics from line
            duration = float(element.get('duration'))
            timeLoss_arr_temp.insert(i, float(element.get('timeLoss'))/duration)
            waitingTime_arr_temp.insert(i, float(element.get('waitingTime'))/duration)
            i += 1
            lineno += 1
        else:                               # average metrics and move on to next time step
            if i == 0:              # no data points for this time step, fill it with previous time step's
                if tstep == 0:
                    timeLoss_arr.insert(tstep, 0.0)
                    waitingTime_arr.insert(tstep, 0.0)
                else:
                    timeLoss_arr.insert(tstep, timeLoss_arr[tstep - 1])
                    waitingTime_arr.insert(tstep, waitingTime_arr[tstep - 1])
            else:
                timeLoss_arr.insert(tstep, average(timeLoss_arr_temp))
                waitingTime_arr.insert(tstep, average(waitingTime_arr_temp))

            t += STEP_SIZE
            tstep += 1

            i = 0
            timeLoss_arr_temp = []
            waitingTime_arr_temp = []
    # add the last element as well
    timeLoss_arr.insert(tstep, average(timeLoss_arr_temp))
    waitingTime_arr.insert(tstep, average(waitingTime_arr_temp))

    # record queue length values
    qlength_arr = []
    tree = ET.parse(QLENGTH_FILE)
    root = tree.getroot()
    tstep = 0
    for element in root.iter('qlength'):
        vals = (element.get('vals'))[1:-1]
        qlens = [float(n) for n in vals.split(", ")]
        qlength_arr.insert(tstep, average(qlens))
        tstep += 1

    timeLoss_all_runs.insert(run, timeLoss_arr)
    waitingTime_all_runs.insert(run, waitingTime_arr)
    qlength_all_runs.insert(run, qlength_arr)
    run += 1

    return


# Desc: Plots all the metrics, averaged across runs, vs time.
# Inputs - title: title of graph
# Outputs - None
def plot_all_metrics(title):

    fig1 = plt.figure()
    fig1.suptitle(title)
    dl, ndl = average_metrics(timeLoss_all_runs)
    plt.subplot(2, 1, 1)
    plt.plot(dl, 'b'), plt.title("Deadlock", fontsize=6)
    plt.xlabel('time steps'), plt.ylabel('timeLoss')
    plt.subplot(2, 1, 2)
    plt.plot(ndl, 'b'), plt.title("Non-deadlock", fontsize=6)
    plt.xlabel('time steps'), plt.ylabel('timeLoss')
    with open(DATAPOINTS_BASE+title+'_tl_'+'dl', 'wb') as fh:
        pickle.dump(dl, fh)
    with open(DATAPOINTS_BASE+title+'_tl_'+'ndl', 'wb') as fh:
        pickle.dump(ndl, fh)

    fig2 = plt.figure()
    fig2.suptitle(title)
    dl, ndl = average_metrics(waitingTime_all_runs)
    plt.subplot(2, 1, 1)
    plt.plot(dl, 'r'), plt.title("Deadlock", fontsize=6)
    plt.xlabel('time steps'), plt.ylabel('waitingTime')
    plt.subplot(2, 1, 2)
    plt.plot(ndl, 'r'), plt.title("Non-deadlock", fontsize=6)
    plt.xlabel('time steps'), plt.ylabel('waitingTime')
    with open(DATAPOINTS_BASE+title+'_wt_'+'dl', 'wb') as fh:
        pickle.dump(dl, fh)
    with open(DATAPOINTS_BASE+title+'_wt_'+'ndl', 'wb') as fh:
        pickle.dump(ndl, fh)

    fig3 = plt.figure()
    fig3.suptitle(title)
    dl, ndl = average_metrics(qlength_all_runs)
    plt.subplot(2, 1, 1)
    plt.plot(dl, 'g'), plt.title("Deadlock", fontsize=6)
    plt.xlabel('time steps'), plt.ylabel('Queue length')
    plt.subplot(2, 1, 2)
    plt.plot(ndl, 'g'), plt.title("Non-deadlock", fontsize=6)
    plt.xlabel('time steps'), plt.ylabel('Queue length')
    with open(DATAPOINTS_BASE+title+'_ql_'+'dl', 'wb') as fh:
        pickle.dump(dl, fh)
    with open(DATAPOINTS_BASE+title+'_ql_'+'ndl', 'wb') as fh:
        pickle.dump(ndl, fh)

    fig4 = plt.figure()
    fig4.suptitle(title)
    run_lengths = [len(waitingTime_all_runs[i]) for i in range(run)]
    plt.plot(run_lengths, 'm')
    plt.xlabel('Live runs'), plt.ylabel('Run length')
    with open(DATAPOINTS_BASE+title+'_runlen', 'wb') as fh:
        pickle.dump(run_lengths, fh)

    fig5 = plt.figure()
    fig5.suptitle(title)
    plt.plot(huber_loss_arr, 'k')
    plt.xlabel('Training Run length'), plt.ylabel('Huber Loss Distribution')
    with open(DATAPOINTS_BASE+title+'_huber_loss', 'wb') as fh:
        pickle.dump(huber_loss_arr, fh)

    fig6 = plt.figure()
    fig6.suptitle(title)
    plt.plot(running_rwrd_arr)
    plt.xlabel('Training Run length'), plt.ylabel('Reward Distribution')
    with open(DATAPOINTS_BASE+title+'_run_rwrd', 'wb') as fh:
        pickle.dump(running_rwrd_arr, fh)

    fig7 = plt.figure()
    fig7.suptitle(title)
    plt.plot(running_avg_rwrd_arr)
    plt.xlabel('Training Run length'), plt.ylabel('Average Reward Distribution')
    with open(DATAPOINTS_BASE+title+'_avg_rwrd', 'wb') as fh:
        pickle.dump(running_rwrd_arr, fh)
    
    plt.show(block=False)

    return


# Average given metric across runs
def average_metrics(metric_all_runs):

    metric_dl_all_runs = []
    metric_ndl_all_runs = []

    # Segregate runs as Deadlock or otherwise
    for i in range(run):
        if len(metric_all_runs[i]) > DEADLOCK_THRESH:
            metric_dl_all_runs.append(metric_all_runs[i])
        else:
            metric_ndl_all_runs.append(metric_all_runs[i])

    avg_metric_dl = [0.0]
    if len(metric_dl_all_runs) > 0:
        minlen_dl = len(metric_dl_all_runs[0])
        for i in range(len(metric_dl_all_runs)):
            minlen_dl = min(minlen_dl, len(metric_dl_all_runs[i]))

        avg_metric_dl = [0.0] * minlen_dl
        for i in range(len(metric_dl_all_runs)):
            avg_metric_dl = [avg_metric_dl[j] + metric_dl_all_runs[i][j] for j in range(minlen_dl)]

        avg_metric_dl = [avg_metric_dl[j]/len(metric_dl_all_runs) for j in range(minlen_dl)]

    avg_metric_ndl = [0.0]
    if len(metric_ndl_all_runs) > 0:
        minlen_ndl = len(metric_ndl_all_runs[0])
        for i in range(len(metric_ndl_all_runs)):
            minlen_ndl = min(minlen_ndl, len(metric_ndl_all_runs[i]))

        avg_metric_ndl = [0.0] * minlen_ndl
        for i in range(len(metric_ndl_all_runs)):
            avg_metric_ndl = [avg_metric_ndl[j] + metric_ndl_all_runs[i][j] for j in range(minlen_ndl)]

        avg_metric_ndl = [avg_metric_ndl[j]/len(metric_ndl_all_runs) for j in range(minlen_ndl)]

    return avg_metric_dl, avg_metric_ndl


# Calculates average of a list
def average(lst):
    return sum(lst) / len(lst)

def huber_loss_record(huber_loss):
	huber_loss_arr.append(huber_loss)

def running_reward_record(running_rwrd):
	running_rwrd_arr.append(running_rwrd)

def running_avg_reward_record(running_avg_rwrd):
	running_avg_rwrd_arr.append(running_avg_rwrd)
