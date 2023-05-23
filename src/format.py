## Imports
from nlb_tools.nwb_interface import NWBDataset
import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler


def picklize(dataset:str):
    if dataset == 'MC_Maze_S':
        data = NWBDataset("./data/000140/sub-Jenkins/", "*train", split_heldout=False)
        with open('data/pickle/mc_maze_s_train.pickle', 'wb') as f:
            pickle.dump(data, f)
        data = NWBDataset("./data/000140/sub-Jenkins/", "*test", split_heldout=False)
        with open('data/pickle/mc_maze_s_test.pickle', 'wb') as f:
            pickle.dump(data, f)
        return True
    else :
        return False
    
def load_data(dataset:str):
    if dataset == 'MC_Maze_S':
        with open('data/pickle/mc_maze_s_train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        with open('data/pickle/mc_maze_s_test.pickle', 'rb') as f:
            test_data = pickle.load(f)
        return train_data.data, test_data.data
    else :
        return None, None