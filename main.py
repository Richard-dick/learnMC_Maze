## import package
import numpy as np
import yaml
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

CONFIG_PATH = 'config/mc_maze.yaml'

OPTIMIZE = False

if __name__ == '__main__':
    if not PICKLIZED:
        picklize("MC_Maze")
    train_data, val_data = load_data("MC_Maze", DIV_FRAC)
    
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    
    for model_name in MODEL:
        for target_var in VAR_GROUPS:
            model_config = {key:config[model_name][key] for key in ['general','opt',target_var]}
            train_var, val_var = restrict_data(train_data, val_data, target_var)
            model, HyperParams = train(train_var['spikes'], train_var['behavior'], train_var['condition'], model_name, model_config, OPTIMIZE)
            
            val_data['estimate'] = model.predict(val_data['spikes'])
            
            model.evaluate()
    
