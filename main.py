## import package
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

import pandas as pd
# pd.set_option('display.min_rows',5000)
# pd.set_option('display.max_rows',5000)
# pd.set_option('display.max_columns',None)

## using my func
from src.format import picklize, load_data, restrict_data
from src.model import train

PICKLIZED = True

DIV_FRAC = 0.2

VAR_GROUPS:list = ['pos','vel']

MODEL = ['gru', 'ffn']

if __name__ == '__main__':
    if not PICKLIZED:
        picklize("MC_Maze")
    train_data, val_data = load_data("MC_Maze", DIV_FRAC)
    
    for target_var in VAR_GROUPS:
        train_var, train_var = restrict_data(train_data, val_data, target_var)
        
        for model_name in MODEL:
            model = train(train_data, model_name)
    
    
    
    # print(train_data["hand_pos"])
    # print(type(test_data))
    