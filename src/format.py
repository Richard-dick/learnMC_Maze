## Imports
from nlb_tools.nwb_interface import NWBDataset
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

target_list = ["spikes", "pos", "condition"]

def picklize(dataset_name:str):
    if dataset_name == 'MC_Maze_S':
        dataset = NWBDataset("./data/000140/sub-Jenkins/", "*train", split_heldout=False)
        trial_info = dataset.trial_info
        trial_data = dataset.data
        print(trial_data["spikes"])
        exit(0)
        for i in range(100):
            print(trial_info["start_time"][i]+"----->"+trial_info["end_time"][i])
            
        print(type(trial_info))
        print(trial_info["start_time"][0])
        exit(0)
        
        data = dict()
        # * 开始转成需要的格式
        with open('data/pickle/mc_maze_s_train.pickle', 'wb') as f:
            
            
            
            pickle.dump(data, f)
        dataset = NWBDataset("./data/000140/sub-Jenkins/", "*test", split_heldout=False)
        with open('data/pickle/mc_maze_s_test.pickle', 'wb') as f:
            pickle.dump(data, f)
        return True
    else :
        return False
    
def load_data(dataset_name:str):
    if dataset_name == 'MC_Maze_S':
        with open('data/pickle/mc_maze_s_train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        with open('data/pickle/mc_maze_s_test.pickle', 'rb') as f:
            test_data = pickle.load(f)
        return train_data.data, test_data.data
    else :
        return None, None