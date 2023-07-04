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
        start = onset - pd.to_timedelta("512ms")
        end = onset + pd.to_timedelta("511ms")
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

def store_results(MSE, behavior, behavior_estimate, HyperParams, Results, var_group):
    Results[var_group] = dict()
    Results[var_group]['MSE'] = MSE
    Results[var_group]['behavior'] = behavior
    Results[var_group]['behavior_estimate'] = behavior_estimate
    Results[var_group]['HyperParams'] = HyperParams.copy()

    
    
def save_data(Results, run_name):
    # If the 'results' directory doesn't exist, create it.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save Results as .pickle file.
    with open('results/' + run_name + '.pickle','wb') as f:
        pickle.dump(Results,f)