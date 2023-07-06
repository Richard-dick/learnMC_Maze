## import package
import numpy as np
import yaml
import pandas as pd
# pd.set_option('display.min_rows',5000)
# pd.set_option('display.max_rows',5000)
# pd.set_option('display.max_columns',None)

## using my func
from src.format import picklize, load_data, restrict_data, store_results, save_data
from src.model import train


PICKLIZED = True

DIV_FRAC = 0.2

VAR_GROUPS:list = ['pos']
# MODEL = 'ffn'

CONFIG_PATH = 'config/mc_maze.yaml'

OPTIMIZE = False

# RUN = 'future_trace'
RUN = "32-8-trace"

Results = dict()

TRACE = 10

if __name__ == '__main__':
    if not PICKLIZED:
        picklize("MC_Maze")
        print("picklize over!!")
        exit(0)
    train_data, val_data = load_data("MC_Maze", DIV_FRAC)
    
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        
    for target_var in VAR_GROUPS:
        model_config = {key:config['ffn'][key] for key in ['general','opt',target_var]}
        train_var, val_var = restrict_data(train_data, val_data, target_var)
        model, HyperParams = train(train_var['spikes'], train_var['behavior'], train_var['condition'], model_config, OPTIMIZE)
        
        val_var['estimate'] = model.predict(val_var['spikes'])
        
        # Evaluate performance, excluding the first tau samples for
        # which sufficient spiking history did not exist for all methods.
        
        MSE = model.evaluate(val_var['behavior'], val_var['estimate'], visulize=True, save_dir=RUN)
        
        print('{} MSE: {}'.format(target_var, MSE))

        # Store performance in 'Results' dictionary.
        store_results(MSE, val_var['behavior'], val_var['estimate'], HyperParams, Results, target_var)

# Save results.
save_data(Results, RUN)
