import configparser
import pickle
import re
from os.path import join as oj
from tracemalloc import stop
import h5py
import numpy as np
from scipy import interpolate

config = configparser.ConfigParser()
config.read("../config.ini")

def get_charge_discharge_(I, t, offset=10):
    '''
    Sees when the battery is undergoing charging, or discharging based on the 
    current. Return the charge in the battery as a function of time
    '''
    length_dataset = len(I)

    assert length_dataset > offset + 1 #Require that the dataset has a certain length if not, the model
    #will most likely perform poorly
    charging_current = False
    if I[offset] > 0:
        #If the current is positive, the battery is being charged
        charging_current = True
        dt = np.mean(t[1:]-t[0:-1]) #Get the average time difference between two point
        return np.cumsum(np.multiply(I,dt)) , charging_current
    else:
        #The charge as a function of time is equal to the integral of 
        #the charge over time 
        dt = np.mean(t[1:]-t[0:-1])
        #We multiply by -1, as the model cares only about the absolute value of the current 
        return -np.cumsum(np.multiply(I,dt)) , charging_current
            
    
def transform_arbin_data_to_dict(data, cell=''):
    '''
    Transforms the input ARBIN data into the correct dictionary format such 
    that we can train Lauras model 
    '''
    cycle_dict = {}
    Qd_mean = []
    for j in range(0, len(data),2): #We loop over all cycles for a given channel/cell 
        try:
            cycle = "Cycle_" + str(j + 1)
            I1 = np.array(data[cycle]["I"])
            V1 = np.array(data[cycle]["V"])
            t1 = np.array(data[cycle]["t"])
            
            cycle_prev = "Cycle_" + str(j)
            Iprev = np.array(data[cycle_prev]["I"])
            Vprev = np.array(data[cycle_prev]["V"])
            tprev = np.array(data[cycle_prev]["t"])
            
            #We define cycles pairwise 
            # as one cycle is defined as either a charging, or a discharging cycle 
            #Is a discharging process 
        except:
            continue
        #We do not know a_priori if the cell is charging or discharging...
        Q, is_charging = get_charge_discharge_(I1,t1)
        if is_charging:
            Qc = Q
            tc = t1
            td = tprev
            Qd, _ = get_charge_discharge_(Iprev,tprev)
            
            # Split the process into a charging, and a 
            # discharging process with corresponding voltages, currents, and charge
            
            cd = {'Ic': I1, 'Id' :  Iprev, 'Qc': Qc, 'Qd': Qd, 
                   'Vc':V1,'Vd': Vprev,  'td':td, 'tc': tc}
        else:
            Qd = Q
            td = t1
            tc = tprev
            Qc, _ = get_charge_discharge_(Iprev,tprev)
            cd = {'Ic': Iprev, 'Id' : I1, 'Qc': Qc, 'Qd': Qd, 
                   'Vc':Vprev,'Vd' : V1,  'td':td, 'tc': tc}
        
        #I do not know how to calculate the capacity_curve referred to in the paper
        # So for now I just assume that the capacity, is equal to the capacity when the battery is charged fully
        Qd_mean.append(Qd[-1])
         #The charge policy, meaning how we charged, and discharged the cell
         #This is not relevant for how we train the model, so for now we just set some value 
        
        #We turn them in
        cycle_dict[str(int(j/2))] = cd
    policy = "3.6C(80%)-3.6C"
    #We also need a dictionary over the summary of the charging and discharging protocols 

    #Remember this dicitonary is only for testing with Leons data ...
    "cell10", "cell19", "cell6", "cell7"
    lifetime = {"cell10": 118, "cell19": 128, "cell6" : 236, "cell7" : 194}
    cycle_life = lifetime[cell]

    cell_dict = {'cycle_life': cycle_life, 
                 'charge_policy':policy, 
                 'summary': {'QC' : Qd_mean[0:cycle_life]}, 
                 'cycles': cycle_dict}
    return cell_dict

def load_data_all_channels(abs_file_path):
    '''

    Input = absolute filepath to where the arbin data is stored
    RETURNS a battery dictionary containing all data from a set of cell experiments.
    The dictionary is compatible with the rest of the ML framework
    
    '''
    #For testing with the Data given by Leon with the 14 cells, only these cells 
    #reached 80 % capacity within their respective cycles
    #This code is to be removed later ... 

    cells_reached_80 = ["cell10", "cell19", "cell6", "cell7"]
    bat_dict = {} #Contains the data from all the cycles for the different channels
    with h5py.File(abs_file_path, "r") as f:
        #Loop over all the channels/cells 
        for CH_Nr in list(f.keys()):
            if CH_Nr in cells_reached_80:

                data_channel = f[CH_Nr]["split"] #We get the cycling data from a specific channel 
                chn_data_dict = transform_arbin_data_to_dict(data_channel, CH_Nr) #We transform the data such that 
                bat_dict.update({CH_Nr: chn_data_dict})
            #it can be used to train the LSTM model 
    
    return bat_dict
        
def get_split(my_len, seed=42):
    """ split a given number of samples reproducibly into training, validation and test

    Args:
        my_len (int]): number of samples to split
        seed (int, optional): [description]. Defaults to 42.

    Returns:
        [type]: tuple of indexes to allocate reproducibly to 
    """

    split_idxs = np.arange(my_len)
    np.random.seed(seed)
    np.random.shuffle(split_idxs)

    num_train, num_val, = (
        int(my_len * 0.5),
        int(my_len * 0.25),
    )
    num_test = my_len - num_train - num_val
    train_idxs = split_idxs[:num_train]
    val_idxs = split_idxs[num_train:num_train + num_val]
    test_idxs = split_idxs[-num_test:]
    return train_idxs, val_idxs, test_idxs


def get_qefficiency(battery, cycle):
    """for a given battery data and cycle, calculate the coloumbic efficiency

    Args:
        battery ([type]): [description]
        cycle ([type]): [description]

    Returns:
        [type]: [description]
    """
    #Gets the voltage and charge as a function of time when the battery is charging
    _, cq_curve = get_capacity_curve(battery, cycle, is_discharge=False)
    
    #Gets the voltage and charge as a function of time when the battery is dicharging
    dv_curve, dq_curve = get_capacity_curve(battery, cycle, is_discharge=True)

    dv_curve, dq_curve = dv_curve[dv_curve > 2.75], dq_curve[dv_curve > 2.75]
    #Take only values where the voltage is larger than 2.005V

    dq_curve_abs = dq_curve.max()  
    cv_curve_abs = cq_curve.max() 
    # return dq_curve_abs, cv_curve_abs, dq_curve_abs / cv_curve_abs

    #Returns the ratio between the maximum charge on the battery during discharge,
    #and during charging respecitvely
    return -1, -1, dq_curve_abs / cv_curve_abs


def get_mean_voltage_difference(battery, cycle):
    """calculate overpotential for a given battery and cycle
    TODO 
    Find an appropriate lower bound for the voltage for a given 
    battery, and make it automatic 
    """
    cv_curve, _ = get_capacity_curve(battery, cycle, is_discharge=False)
    dv_curve, dq_curve = get_capacity_curve(battery, cycle, is_discharge=True)
    dv_curve, dq_curve = dv_curve[dv_curve > 2.7], dq_curve[dv_curve > 2.7]

    # start_cap = cq_curve.max()

    return cv_curve.mean() - dv_curve.mean()


def get_capacity_curve(cell, cycle, is_discharge):
    """
    Returns the charge inside the battery, when it is charged or discharged
    """

    # We get the voltage, and charge on the battery during charging 
    # and discharging processes respectively 
    
    if is_discharge:
        q_curve = cell["cycles"][str(cycle)]["Qd"]
        v_curve = cell["cycles"][str(cycle)]["Vd"]
    else:
        #Or start the count when the cell is charging 
        q_curve = cell["cycles"][str(cycle)]["Qc"]
        v_curve = cell["cycles"][str(cycle)]["Vc"]
    
    return (v_curve, q_curve)

def smooth_x(
    x,
    y,
    num_points=10,
):
    """Smoothes input for training by averaging over the num_points points.

    Args:
        x Numpy Array: Capacity curves for all batteries
        y ([type]): 1D array indicating the lif
        num_points (int, optional): number of points over which to average. Defaults to 10.

    Returns:
        Numpy Array: [description]
    """
    x_smoothed = x.copy()

    for i in range(len(x)):
        x_smoothed[i, num_points // 2:-num_points // 2 + 1] = np.convolve(
            x[i], np.ones((num_points)) / num_points, mode="valid")
    return x_smoothed


def get_capacity_spline(cell, cycle):
    """
    splines the voltage capacity curve
    """
    #Get charge/discharge curve, and voltage curve as a the cell goes from 
    # fully charged -> discharged, or discharged -> fully charged
    
    v_curve, q_curve = get_capacity_curve(cell, cycle, is_discharge=True)
    f = interpolate.interp1d(v_curve, q_curve, fill_value="extrapolate")
    # We try to interpolate Q = Q(V)
    points = np.linspace(4.25, 2.75, num=1000) 
    # Points are taken between, the maximum, and minimum voltage of a cell
    # For the data which we recieved from Leon, this is reported as
    # V_max = 4.25, and V_min = 2.75
    spline = f(points)
    spline[np.where(np.isnan(spline))] = 0
    return spline


def scale_x(x, y):
    max_val = 1.1  # nominal capacity for the cells which you are testing (in Ah)
    end_of_life_val = (
        0.8 * 1.1
    )  # batteries are considered dead after 80%. This should be .8*1.1
    x = np.minimum(x, max_val)
    x = np.maximum(x, end_of_life_val)

    x = (x - end_of_life_val) / (max_val - end_of_life_val)

    return x


def remove_outliers(x_in, y):
    x = x_in.copy()

    for i in range(2, x.shape[1]):
        avg = (x[:, i - 1] + x[:, i - 2]) / 2
        too_low = (x[:, i]) / (avg + 0.0001) < 0.80
        too_high = (x[:, i]) / (avg + 0.0001) > (1.10)
        idx = np.where((too_low + too_high) * (i < y))
        x[idx, i] = x[idx, i - 1]
    return x

# If we want to make sure that this works with the Arbin,
# we have to specify which path the data from the Arbin server is sent to 

        
def load_data_single_channel(data_path, CH_Nr = None):
    '''
    Loads data and turns it into the proper format for training 
    Our models 
    The data is structured such that 
    TODO
    Make it such that we loop over all the cells we are testing
    '''
    return 


def get_charge_policy(my_string):

    # charge policy extract from string
    vals = [
        float(x[0] + x[1]) for x in re.findall(r"(\d\.\d+)|(\d+)", my_string)
    ]  # get three  ints or floats from a string,
    return vals


def get_max_life_time(data_dict):
    '''
    Get maximum cell lifetime from an experiment involving several cells 
    Input: data_dict 
    Example 
    data_dict = {'cell10' : {'cell_lifetime': 312, .... },
    'cell11' :  {'cell_lifetime': 417, .... }}
    returns 417
    '''
    max_lifetime = 0
    for bat in data_dict.keys():
        max_lifetime = np.maximum(max_lifetime,
                                  data_dict[bat]["cycle_life"])
    return int(max_lifetime)


def transform_charging(c):
    num_steps = 9
    stop_val = 0.8
    long_c = np.zeros((len(c), num_steps))
    for i in range(num_steps):
        use_first = c[:, -1] > i / num_steps * stop_val
        long_c[:, i] = c[:, 0] * use_first + c[:, 2] * (1 - use_first)
    return long_c


def get_capacity_input(data_dict,
                       num_offset=0,
                       start_cycle=10,
                       stop_cycle=100,
                       use_long_cschedule=False):
    for key in data_dict.keys():
        #We loop over all the individual cells
        #For example key = 'b1c0' which is the cell named b1c0
        bat = data_dict[key]
        
        start_curve = get_capacity_spline(
            bat,
            start_cycle, 
        )

        stop_curve = get_capacity_spline(
            bat,
            stop_cycle,
        )
        
        idxs = np.where(
            1 - (np.isnan(start_curve) + np.isnan(stop_curve))
        )  # the first parts of the cycle don't get modelled well. Ignored because tiny part
        bat["qv_variance"] = np.log(
            (start_curve[idxs] - stop_curve[idxs]).var())
        # We take the variance between the interpolated function, 
        # based on data from the start_cycle
        # and the interpolated function based on the stop_cycle
        bat["c_efficiency"] = (get_qefficiency(bat, stop_cycle)[2] -
                               get_qefficiency(bat, start_cycle)[2])
        #Look at how the mean value of the voltage curve changes from starting cycle to ending cycle 
        bat["v_delta"] = get_mean_voltage_difference(
            bat, stop_cycle) - get_mean_voltage_difference(
                bat, start_cycle)  # always use the fifth cycle

    max_lifetime = get_max_life_time(data_dict) - num_offset

    num_bats = len(data_dict)
    max_lifetime = 236
    x = -1 * np.ones((num_bats, max_lifetime))
    charge_policy = np.zeros((num_bats, 4))
    in_cycle_data = np.zeros((num_bats, 3))

    y = np.zeros(num_bats)
    for i, bat in enumerate(data_dict.keys()):
        x[i, :len(data_dict[bat]["summary"]["QC"]) -
          num_offset] = data_dict[bat]["summary"]["QC"][num_offset:]
        y[i] = data_dict[bat]["cycle_life"]
        first, switch_time, second = get_charge_policy(
            data_dict[bat]["charge_policy"])

        switch_time /= 100

        avg = first * (switch_time / 0.8) + second * (
            1 - switch_time / 0.8)  # batteries charged from .0 until .8 SOC
        charge_policy[i] = first, avg, second, switch_time
        in_cycle_data[i, 0] = data_dict[bat]["qv_variance"]
        in_cycle_data[i, 1] = data_dict[bat]["c_efficiency"]
        in_cycle_data[i, 2] = data_dict[bat]["v_delta"]
    y = y.astype(np.int32)

    # The function, returns the summary of the charging current (x)
    # Lifetime of the battery, defined as cycles to reach 80 % capacity (y)
    # The charging policy for example [3.6C 0.9 3.6C] so [charge SOC discharge]
    # In cycle_data = [qv_variance, coulombic efficiency, v_delta]
    if use_long_cschedule:
        return x, y, charge_policy, in_cycle_data
    return x, y, charge_policy[:, :3], in_cycle_data


def assemble_dataset(x, y, augment, seq_len=50, use_cycle_counter=True):

    xs = []
    ys = []
    add_data_s = []
    for i in range(len(x)):
        for j in range(y[i] - seq_len - 3):

            xs.append(x[i, j:j + seq_len])

            ratio = x[i, j + seq_len] / (x[i, j + seq_len - 1] + 10e-17)

            ys.append((ratio, (y[i] - seq_len - j)))
            if use_cycle_counter:
                add_data_s.append(np.hstack([augment[i], np.log(j + seq_len)]))
            else:
                add_data_s.append(augment[i])
    return np.asarray(xs)[:, :, None], np.asarray(ys), np.asarray(add_data_s)
