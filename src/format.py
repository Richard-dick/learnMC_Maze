## Imports
from nlb_tools.nwb_interface import NWBDataset
import pickle
import numpy as np
import pandas as pd
import os
from src.utils import partition
import copy

target_list = ["spikes", "pos", "condition"]

def picklize(dataset_name:str):
    print("start reading")
    if dataset_name == 'MC_Maze_S':
        dataset = NWBDataset("./data/000140/sub-Jenkins/", "*train", split_heldout=False)
    elif dataset_name == 'MC_Maze':
        dataset = NWBDataset("./data/000128/sub-Jenkins/", "*train", split_heldout=False)     
    else :
        return False
    if not os.path.exists('data/pickle'):
        os.makedirs('data/pickle')
    print("reading over, now transferring")
    trial_info = dataset.trial_info
    trial_data = dataset.data
    # * 开始转成需要的格式
    ## condition
    data = dict()
    data["condition"] = list(trial_info["maze_id"])
    
    pos_list = list()
    spikes_list = list()
    vel_list = list()
    onset_time = list(trial_info["move_onset_time"])
    print("start transpose")
    for idx in range(len(onset_time)):
        print("round: "+ str(idx))
        onset = onset_time[idx]
        start = onset - pd.to_timeBin_Size("549ms")
        end = onset + pd.to_timeBin_Size("450ms")
        pos = np.array(trial_data.loc[start:end, "hand_pos":"hand_pos"]).T
        spikes = np.array(trial_data.loc[start:end, "spikes":"spikes"]).T
        vel = np.array(trial_data.loc[start:end, "hand_vel":"hand_vel"]).T
        pos_list.append(pos)
        spikes_list.append(spikes)
        vel_list.append(vel)
    data["pos"] = pos_list
    data["spikes"] = spikes_list
    data["vel"] = vel_list
    print("transposing over")

    
    if dataset_name == 'MC_Maze_S':
        with open('data/pickle/mc_maze_s_train.pickle', 'wb') as f:
            pickle.dump(data, f)
    elif dataset_name == 'MC_Maze':
        with open('data/pickle/mc_maze_train.pickle', 'wb') as f:
            pickle.dump(data, f)
    
    
    # dataset = NWBDataset("./data/000140/sub-Jenkins/", "*test", split_heldout=False)
    # with open('data/pickle/mc_maze_s_test.pickle', 'wb') as f:
    #     pickle.dump(data, f)
    
def load_data(dataset_name:str, val_frac):
    if dataset_name == 'MC_Maze_S':
        with open('data/pickle/mc_maze_s_train.pickle', 'rb') as f:
            data = pickle.load(f)
        
        return data
    elif dataset_name == 'MC_Maze':
        with open('data/pickle/mc_maze_train.pickle', 'rb') as f:
            data = pickle.load(f)
        train_idx, test_idx = partition(data['condition'], val_frac)
        Train = dict()
        Test = dict()
        for k in data.keys():
            Train[k] = [data[k][i] for i in train_idx]
            Test[k] = [data[k][i] for i in test_idx]
    
        return Train, Test
    else :
        return None, None
    
def restrict_data(Train:np.array, Test:np.array, var_group:str):
    
    # Initialize outputs.
    Train_b = dict()
    Test_b = dict()
    
    # Copy spikes into new dictionaries.
    Train_b['spikes'] = copy.deepcopy(Train['spikes'])
    Train_b['condition'] = copy.deepcopy(Train['condition'])
    Train_b['behavior'] = copy.deepcopy(Train[var_group])
    
    Test_b['spikes'] = copy.deepcopy(Test['spikes'])
    Test_b['condition'] = copy.deepcopy(Test['condition'])
    Test_b['behavior'] = copy.deepcopy(Test[var_group])

    return Train_b, Test_b

def store_results(R2, behavior, behavior_estimate, HyperParams, Results, var_group, Train):

    """
    Store coefficient of determination (R2), ground truth behavior, 
    decoed behavior, and hyperparameters in the Results dictionary
    under key(s) indicating the behavioral variable group(s) these
    results correspond to.

    Inputs
    ------
    R2: 1D numpy array of coefficients of determination

    behavior: list of M x T numpy arrays, each of which contains ground truth behavioral data for M behavioral variables over T times

    behavior_estimate: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times

    HyperParams: dictionary of hyperparameters

    Results: method- and dataset-specific dictionary to store results in

    var_group: string (or list of strings) containing behavioral variable group(s)
        these results are associated with

    Train: dictionary containing trialized neural and behavioral data in training set
        This only gets used to help determine which R2 values go with which behavioral variables.
   
    """

    # Store R2, behavior, decoded behavior, and HyperParams in Results with the appropriate key.
    if isinstance(var_group,str):
        Results[var_group] = dict()
        Results[var_group]['R2'] = R2
        Results[var_group]['behavior'] = behavior
        Results[var_group]['behavior_estimate'] = behavior_estimate
        Results[var_group]['HyperParams'] = HyperParams.copy()
    elif isinstance(var_group,list):
        i = 0
        for v in var_group:
            m = Train[v][0].shape[0]
            Results[v] = dict()
            Results[v]['R2'] = R2[i:i+m]
            Results[v]['behavior'] = [b[i:i+m,:] for b in behavior]
            Results[v]['behavior_estimate'] = [b[i:i+m,:] for b in behavior_estimate]
            Results[v]['HyperParams'] = HyperParams.copy()
            i += m
    else:
        raise Exception('Unexpected type for var_group.')
    
    
def save_data(Results, run_name):

    """
    Save decoding results.

    Inputs
    ------
    Results: dictionary containing decoding results

    run_name: filename to use for saving results (without .pickle extension)
   
    """

    # If the 'results' directory doesn't exist, create it.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save Results as .pickle file.
    with open('results/' + run_name + '.pickle','wb') as f:
        pickle.dump(Results,f)