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

SPIKE_GROUPS:list = ["spikes", "PMd_spikes", "MI_spikes"]
VAR_GROUPS:list = ['target_pos', 'pos']
# MODEL = 'ffn'

CONFIG_PATH = 'config/mc_maze.yaml'

RUN = "32-8-trace"

Results = {var:dict() for var in VAR_GROUPS}

if __name__ == '__main__':
    # 获取数据
    if not PICKLIZED:
        # picklize("MC_Maze_all")
        picklize("MC_Maze_sep")
        print("picklize over!!")
        exit(0)
    train_data, val_data = load_data("MC_Maze_sep", DIV_FRAC)

    # 载入配置
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        
    # 遍历目标
    for target_var in VAR_GROUPS:
        # 取出对应目标行为的 config
        model_config = {key:config['ffn'][key] for key in ['general','opt', target_var]}
        # 遍历神经活动类型:
        for spike_type in SPIKE_GROUPS:
            # 获取本次训练需要的数据
            train_var, val_var = restrict_data(train_data, val_data, spike_type, target_var)
            model, HyperParams = train(train_var, model_config)
            # 预测结果
            val_var['estimate'] = model.predict(val_var['spikes'])
            # 分析结果
            MSE = model.evaluate(val_var['behavior'], val_var['estimate'], visulize=True, save_dir=RUN)
            
            print('The MSE about {} using {}: {}'.format(target_var, spike_type, MSE))

            # Store performance in 'Results' dictionary.
            store_results(MSE, val_var['behavior'], val_var['estimate'], HyperParams, Results[target_var], spike_type)

    # Save results.
    save_data(Results, RUN)
