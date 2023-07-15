## Imports
from nlb_tools.nwb_interface import NWBDataset
import pickle
import numpy as np
import pandas as pd
import os
from src.utils import partition
import copy

pd.set_option('display.max_columns', None)

def picklize(dataset_name:str):
    print("start reading")
    if dataset_name == 'MC_Maze_sep':
        dataset = NWBDataset("./data/000128/sub-Jenkins/", "*train", split_heldout=False)
    elif dataset_name == 'MC_Maze_all':
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
    PMd_spikes_list = list()
    MI_spikes_list = list()
    vel_list = list()
    target_pos_list = list()
    onset_time = list(trial_info["move_onset_time"])
    activate_target = list(trial_info["active_target"])
    target_pos = list(trial_info["target_pos"])
    success = list(trial_info["success"])
    print("start transpose")
    for idx in range(len(onset_time)):
        print("round: "+ str(idx))
        onset = onset_time[idx]
        start = onset - pd.to_timedelta("512ms")
        end = onset + pd.to_timedelta("511ms")
        pos = np.array(trial_data.loc[start:end, "hand_pos":"hand_pos"]).T
        spikes = np.array(trial_data.loc[start:end, "spikes":"spikes"]).T
        # print(spikes.shape)
        # exit(0)
        
        vel = np.array(trial_data.loc[start:end, "hand_vel":"hand_vel"]).T
        target_pos_item = target_pos[idx][activate_target[idx]]
        pos_list.append(pos)
        spikes_list.append(spikes)
        PMd_spikes_list.append(spikes[0:92,:])
        MI_spikes_list.append(spikes[92:,:])
        vel_list.append(vel)
        target_pos_list.append(target_pos_item)
    data['success'] = success
    data["pos"] = pos_list
    data["spikes"] = spikes_list
    data["PMd_spikes"] = PMd_spikes_list
    data["MI_spikes"] = MI_spikes_list
    data["vel"] = vel_list
    data['target_pos'] = target_pos_list
    print("transposing over")

    
    if dataset_name == 'MC_Maze_sep':
        with open('data/pickle/mc_maze_sep.pickle', 'wb') as f:
            pickle.dump(data, f)
    elif dataset_name == 'MC_Maze_all':
        with open('data/pickle/mc_maze_all_train.pickle', 'wb') as f:
            pickle.dump(data, f)
            
    
def load_data(dataset_name:str, val_frac):
    if dataset_name == 'MC_Maze_sep':
        with open('data/pickle/mc_maze_sep.pickle', 'rb') as f:
            data = pickle.load(f)
        train_idx, test_idx = partition(data['condition'], val_frac)
        Train = dict()
        Test = dict()
        for k in data.keys():
            Train[k] = [data[k][i] for i in train_idx]
            Test[k] = [data[k][i] for i in test_idx]
    
        return Train, Test
        
    elif dataset_name == 'MC_Maze_all':
        with open('data/pickle/mc_maze_all_train.pickle', 'rb') as f:
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
    
def restrict_data(Train:np.array, Test:np.array, spike_type:str, var_group:str):
    
    Train_b = dict()
    Test_b = dict()
    
    # Copy spikes into new dictionaries.
    Train_b['spikes'] = copy.deepcopy(Train[spike_type])
    Train_b['behavior'] = copy.deepcopy(Train[var_group])
    
    Test_b['spikes'] = copy.deepcopy(Test[spike_type])
    Test_b['behavior'] = copy.deepcopy(Test[var_group])
    
    if var_group == 'target_pos':
        Train_b['success'] = copy.deepcopy(Train['success'])
        Test_b['success'] = copy.deepcopy(Test['success'])

    return Train_b, Test_b

def store_results(MSE, behavior, behavior_estimate, HyperParams, Results, spikes_type):
    # Results[spikes_type] = dict()
    Results[spikes_type]['MSE'] = MSE
    Results[spikes_type]['behavior'] = behavior
    Results[spikes_type]['behavior_estimate'] = behavior_estimate
    Results[spikes_type]['HyperParams'] = HyperParams.copy()

    
    
def save_data(Results, run_name):
    # If the 'results' directory doesn't exist, create it.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save Results as .pickle file.
    with open('results/' + run_name + '.pickle','wb') as f:
        pickle.dump(Results,f)