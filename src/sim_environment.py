# Acts as the environment for the RL problem.
# Uses TraCI to monitor/control SUMO simulations. Takes the given action provided and outputs the Next state and Reward.

# Depends on: plot_metrics.py

from sumolib import checkBinary
import traci
import plot_metrics
import array
import os
import platform

T_SWITCH = 12                                               # Intersection switching time
T_TRANS = 4                                                 # Signal transition time when changing sides
OCC_TH_HIGH = 70.0                                          # Detector occupancy high threshold for jamming
OCC_TH_LOW = 30.0                                           # Detector occupancy low threshold for jamming
NUM_INTN_SIDES = 15                                         # No. of intersection sides
NUM_LANES = [3, 2, 3, 2, 3, 2, 3, 2, 1, 1, 1, 1, 2, 1, 2]   # No. of lanes at intersection sides

TL_IDS = ["gneJ20", "gneJ18", "151147259", "gneJ69"]        # Intersection traffic light IDs

# Intersection phases
TL_PHASES = {'N1_G': 0, 'N1_Y': 1, 'N2_G': 2, 'N2_Y': 3, 'N3_G': 4, 'N3_Y': 5, 'N4_G': 6, 'N4_Y': 7,
             'N5_G': 0, 'N5_Y': 1, 'N6_G': 2, 'N6_Y': 3, 'N7_G': 4, 'N7_Y': 5, 'N8_G': 6, 'N8_Y': 7,
             'N9_G': 0, 'N9_Y': 1, 'N10_G': 2, 'N10_Y': 3, 'N11_G': 4, 'N11_Y': 5, 'N12_G': 6, 'N12_Y': 7,
             'N13_G': 0, 'N13_Y': 1, 'N14_G': 2, 'N14_Y': 3, 'N15_G': 4, 'N15_Y': 5}

QLEN_FILE = "../scripts/qlengths.xml"
qlenfile = open(QLEN_FILE, 'w')

sumoBinary = checkBinary('sumo-gui')                        # mode of SUMO

intn_prev_action = [1, 5, 9, 13]                            # previous action taken at the intersections

first_act_of_run = 0                                        # flag for occurrence of first action of a run
skip_time = 0                                               # time spent in initial randomization
t = 0

curr_state = array.array('f', [0] * NUM_INTN_SIDES)
prev_state = array.array('f', [0] * NUM_INTN_SIDES)


# Desc: Starts a new simulation run by generating a random routes file. Also reinitializes the detector counts etc.
# Inputs - run: run number, 0 implies first run
# Outputs - None
def start_new_run(run):

    global curr_state, prev_state, intn_prev_action, first_act_of_run, skip_time, qlenfile

    curr_state = array.array('f', [0] * NUM_INTN_SIDES)
    prev_state = array.array('f', [0] * NUM_INTN_SIDES)
    intn_prev_action = [1, 5, 9, 13]
    first_act_of_run = 0
    skip_time = 0

    # Setup plot_metrics for a new set of runs
    if run == 0:
        plot_metrics.init(T_TRANS + T_SWITCH)

    if platform.system() == 'Windows':
        # Generate a new random routes file
        os.system("python \"%SUMO_HOME%/tools/randomTrips.py\" -n ../scripts/txmap.net.xml --trip-attributes=\"type=\\\"light_norm_heavy\\\"\" "
            "-a ../scripts/txmap.add.xml -r ../scripts/txmap.rou.xml -e 12000 -p 0.75 --binomial=5 -L")
        # Delete unwanted alt route file
        os.system("del \"../scripts/txmap.rou.alt.xml\"")
        # Delete unwanted trips file
        os.system("del trips.trips.xml")
    elif platform.system() == 'Linux':
        # Generate a new random routes file
        os.system("python \"$SUMO_HOME/tools/randomTrips.py\" -n ../scripts/txmap.net.xml --trip-attributes=\"type=\\\"light_norm_heavy\\\"\" "
            "-a ../scripts/txmap.add.xml -r ../scripts/txmap.rou.xml -e 12000 -p 0.75 --binomial=5 -L")
        # Delete unwanted alt route file
        os.system("rm \"../scripts/txmap.rou.alt.xml\"")
        # Delete unwanted trips file
        os.system("rm trips.trips.xml")

    # Start SUMO and connect traCI to it
    traci.start([sumoBinary, "-c", "../scripts/txmap.sumocfg", "--gui-settings-file", "../scripts/guisettings.xml",
                 "--start", "--quit-on-end", "--no-warnings"])

    qlenfile = open(QLEN_FILE, 'w')
    qlenfile.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>" + "\n\n")
    qlenfile.write("<qlengths>" + '\n')
    qlenfile.flush()

    return


# Desc: Gives Green signal to the given intersection side.
# Inputs - a: The intersection side to be given Green signal to, 1<=a<=15. If a = 0, no action is taken
# Outputs - A dictionary containing 2 elements
#               'rwd': Reward; Returns -100 if the current simulation is over
#               'next_state': 15-element array containing the no. of vehicles waiting at the intersections in new state
def take_action(a):

    global curr_state, prev_state, intn_prev_action, first_act_of_run, skip_time, t

    # randomization phase, no control; let static TL logic of SUMO run
    if a == 0:
        for i in range(T_TRANS + T_SWITCH):
            traci.simulationStep()
        curr_state = get_current_state()
        R = calculate_reward(curr_state, prev_state, a)
        skip_time += T_TRANS + T_SWITCH

        # all vehicles left simulation; current run over
        if traci.simulation.getMinExpectedNumber() == 0:
            traci.close()
            qlenfile.write("</qlengths>")
            qlenfile.close()
            plot_metrics.record_metrics_of_run(skip_time)
            return {'rwd': -100, 'next_state': curr_state}

        prev_state = curr_state
        return {'rwd': R, 'next_state': curr_state}
    # randomization phase over
    elif first_act_of_run == 0:
        t = 0                       # time counting starts from here
        first_act_of_run = 1

    # Intersection being controlled
    if 1 <= a <= 4:
        intn = 0
    elif 5 <= a <= 8:
        intn = 1
    elif 9 <= a <= 12:
        intn = 2
    else:
        intn = 3

    # if action same as previous for this intersection
    if a == intn_prev_action[intn]:
        traci.trafficlight.setPhase(TL_IDS[intn], TL_PHASES['N' + str(a) + '_G'])   # Continue Green during transition
        for i in range(T_TRANS):
            traci.simulationStep()
    # different action
    else:
        traci.trafficlight.setPhase(TL_IDS[intn], TL_PHASES['N' + str(intn_prev_action[intn]) + '_Y'])   # Give Yellow during transition
        for i in range(T_TRANS):
            traci.simulationStep()

    # Give Green and wait for T_SWITCH secs
    traci.trafficlight.setPhase(TL_IDS[intn], TL_PHASES['N' + str(a) + '_G'])
    for i in range(T_SWITCH):
        traci.simulationStep()

    curr_state = get_current_state()

    t += T_TRANS + T_SWITCH
    qlenfile.write('\t' + "<qlength vals=\"" + str(curr_state.tolist()) + "\" t=\"" + str(t) + "\"/>" + '\n')
    qlenfile.flush()

    # all vehicles left simulation; current run over
    if traci.simulation.getMinExpectedNumber() == 0:
        traci.close()
        qlenfile.write("</qlengths>")
        qlenfile.close()
        plot_metrics.record_metrics_of_run(skip_time)
        return {'rwd': -100, 'next_state': curr_state}

    R = calculate_reward(curr_state, prev_state, a)

    prev_state = curr_state
    intn_prev_action[intn] = a

    return {'rwd': R, 'next_state': curr_state}


# Desc: Returns the detector occupancies at the intersections currently.
# Inputs - None
# Outputs - state: 15-element array containing the detector occupancies at the intersections currently
def get_current_state():

    state = array.array('f', [0] * NUM_INTN_SIDES)

    for i in range(NUM_INTN_SIDES):
        n = 0.0
        for j in range(NUM_LANES[i]):
            n = max(n, traci.lanearea.getLastStepOccupancy("e2det_N"+str(i+1)+"_"+str(j)))

        state[i] = n

    return state


# Desc: Calculates the reward when transitioning states.
# Inputs - curr_occ: Current state of congestion
#          prev_occ: Previous state of congestion
#          a: Action taken for going from prev to curr state
# Outputs - Reward: Reward
def calculate_reward(curr_occ, prev_occ, a):

    Reward = 0

    # static signalling; reward doesn't matter
    if a == 0:
        return 0

    if a < 5:
        a_space = [1, 2, 3, 4]
        intn = 0
    elif a < 9:
        a_space = [5, 6, 7, 8]
        intn = 1
    elif a < 13:
        a_space = [9, 10, 11, 12]
        intn = 2
    else:
        a_space = [13, 14, 15]
        intn = 3

    if curr_occ[a - 1] < OCC_TH_LOW and prev_occ[a - 1] < OCC_TH_LOW:
        Reward = -40
    elif curr_occ[a - 1] >= OCC_TH_LOW and prev_occ[a - 1] < OCC_TH_LOW:
        Reward = -20
    elif curr_occ[a - 1] < OCC_TH_LOW and (OCC_TH_LOW <= prev_occ[a - 1] < OCC_TH_HIGH):
        prev_occ_arr = []
        for a_temp in a_space:
            prev_occ_arr.append(prev_occ[a_temp - 1])
        prev_occ_max = max(prev_occ_arr)
        if prev_occ_max == prev_occ[a - 1]:
            Reward = 10
        else:
            Reward = 5
    elif curr_occ[a - 1] >= OCC_TH_LOW and (OCC_TH_LOW <= prev_occ[a - 1] < OCC_TH_HIGH):
        prev_occ_arr = []
        for a_temp in a_space:
            prev_occ_arr.append(prev_occ[a_temp - 1])
        prev_occ_max = max(prev_occ_arr)
        if prev_occ_max == prev_occ[a - 1]:
            Reward = 20
        else:
            Reward = 15
    elif curr_occ[a - 1] < OCC_TH_LOW and prev_occ[a - 1] >= OCC_TH_HIGH:
        prev_occ_arr = []
        for a_temp in a_space:
            prev_occ_arr.append(prev_occ[a_temp - 1])
        prev_occ_max = max(prev_occ_arr)
        if prev_occ_max == prev_occ[a - 1]:
            Reward = 30
        else:
            Reward = 25
    elif curr_occ[a - 1] >= OCC_TH_LOW and prev_occ[a - 1] >= OCC_TH_HIGH:
        Reward = 40

    # Add dependency related reward
    N1 = curr_occ[0]; N2 = curr_occ[1]; N3 = curr_occ[2]; N4 = curr_occ[3];
    N5 = curr_occ[4]; N6 = curr_occ[5]; N7 = curr_occ[6]; N8 = curr_occ[7];
    N9 = curr_occ[8]; N10 = curr_occ[9]; N11 = curr_occ[10]; N12 = curr_occ[11];
    N13 = curr_occ[12]; N14 = curr_occ[13]; N15 = curr_occ[14];
    dep_rwd_penality = 5
    if intn == 0:
        if(a==1 and N1>=OCC_TH_HIGH and (N5>=OCC_TH_HIGH or N13>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==2 and N2>=OCC_TH_HIGH and (N5>=OCC_TH_HIGH or N13>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==3 and N3>=OCC_TH_HIGH and (N13>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==4 and N4>=OCC_TH_HIGH and (N5>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
    elif intn == 1:
        if(a==5 and N5>=OCC_TH_HIGH and (N10>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==6 and N6>=OCC_TH_HIGH and (N3>=OCC_TH_HIGH or N10>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==7 and N7>=OCC_TH_HIGH and (N3>=OCC_TH_HIGH or N10>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==8 and N8>=OCC_TH_HIGH and (N3>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
    elif intn == 2:
        if(a==9 and N9>=OCC_TH_HIGH and (N8>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==10 and N10>=OCC_TH_HIGH and (N14>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==11 and N11>=OCC_TH_HIGH and (N8>=OCC_TH_HIGH or N14>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==12 and N12>=OCC_TH_HIGH and (N8>=OCC_TH_HIGH or N14>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
    else:
        if(a==13 and N13>=OCC_TH_HIGH and (N9>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==14 and N14>=OCC_TH_HIGH and (N4>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality
        elif(a==15 and N15>=OCC_TH_HIGH and (N4>=OCC_TH_HIGH or N9>=OCC_TH_HIGH)):
            Reward -= dep_rwd_penality

    return Reward


# Desc: Enable/Disable GUI mode of SUMO
# Inputs - mode: 1 = GUI mode, 0 = non-GUI mode
# Outputs - None
def endis_sumo_guimode(mode):

    global sumoBinary

    if mode == 1:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    return
