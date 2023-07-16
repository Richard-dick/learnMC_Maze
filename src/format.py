## Imports
from nlb_tools.nwb_interface import NWBDataset
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.utils import partition
import copy

# pd.set_option('display.max_columns', None)

SPIKE_GROUPS:list = ["spikes", "PMd_spikes", "MI_spikes"]

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
    Results[spikes_type] = dict()
    Results[spikes_type]['MSE'] = MSE
    Results[spikes_type]['behavior'] = behavior
    Results[spikes_type]['behavior_estimate'] = behavior_estimate
    Results[spikes_type]['HyperParams'] = HyperParams.copy()
    
    
def save_data(Results:dict(), run_name, visualize = False):
    # If the 'results' directory doesn't exist, create it.
    if not os.path.exists('results'):
        os.makedirs('results')
        
    save_dir = 'results/' + run_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save Results as .pickle file.
    with open(save_dir + '/res.pickle','wb') as f:
        pickle.dump(Results,f)
        
    # visualize the result
    if visualize:
        # 先画 trace:       
        if not os.path.exists(save_dir + '/trace/'):
            os.makedirs(save_dir +'/trace/')
        else:
            import shutil
            shutil.rmtree(save_dir + '/trace/')
            os.makedirs(save_dir + '/trace/')
        trace = dict()
        
        tmp = Results['pos']['spikes']['behavior']
        trace['ref'] = [b[:,512+15::16] for b in tmp]
        
        for sp in SPIKE_GROUPS:
            trace[sp] = Results['pos'][sp]['behavior_estimate']
        # ref 是459的list(2*1024), 其余是(2*24*8)
        vis_item = np.random.randint(0, len(trace['ref']))    
        x_ref,y_ref = trace['ref'][vis_item]
        # 选取一个实验结果的24条轨迹, 画出来
        print("starting visualizing trace: "+ str(vis_item))
        save_path = save_dir + '/trace/'
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        
        _, plt_num, trace_length = trace['spikes'][0].shape
        
        for i in range(plt_num):
            plt.cla()
            plt.title('The '+str(i)+' trace')
            plt.plot(x_ref, y_ref, "y-", label = "TRACE")
            x,y = trace['ref'][vis_item][0, i:i+trace_length], trace['ref'][vis_item][1, i:i+trace_length]
            plt.plot(x, y, "r-", label = "ref_trace" )
            
            plt.plot(trace['spikes'][vis_item][0,i,:], trace['spikes'][vis_item][1,i,:], "b-", label = "spikes_trace" )
            plt.plot(trace['MI_spikes'][vis_item][0,i,:], trace['MI_spikes'][vis_item][1,i,:], "m-", label = "MI_trace" )
            plt.plot(trace['PMd_spikes'][vis_item][0,i,:], trace['PMd_spikes'][vis_item][1,i,:], "k-", label = "PMd_trace" )
            
            
            plt.xlabel('x(mm)')
            plt.ylabel('y(mm)')
            plt.legend(loc = "best")
            
            plt.savefig(save_path+str(i)+'.png', format = 'png')

        # 再画 pos 图
        if not os.path.exists(save_dir + '/pos/'):
            os.makedirs(save_dir +'/pos/')
        else:
            import shutil
            shutil.rmtree(save_dir + '/pos/')
            os.makedirs(save_dir + '/pos/')
        aim = dict()
        aim['ref'] = Results['target_pos']['spikes']['behavior_estimate']
        for sp in SPIKE_GROUPS:
            aim[sp] = Results['target_pos'][sp]['behavior_estimate']
        
        print("starting visualizing the target_pos")
        save_path = save_dir + '/pos/'
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        
        # 绘制散点图
        for sp in SPIKE_GROUPS:
            plt.cla()
            plt.scatter(aim['ref'][:,0], aim['ref'][:,1], color = 'red', s = 10, label = "ref_pos") 
            plt.scatter(aim[sp][:,0], aim[sp][:,1], color = 'green', s = 5, label = sp + "_pos" )
            # plt.scatter(aim['MI_spikes'][:,0], aim['MI_spikes'][:,1], color = 'magenta', s = 10, label = "MI_pos" )
            # plt.scatter(aim['PMd_spikes'][:,0], aim['PMd_spikes'][:,1], color = 'black', s = 10, label = "PMd_pos" )
            
            plt.xlabel('x(mm)')
            plt.ylabel('y(mm)')
            plt.title('Scatter Plot of Aim' + '(' + sp +')')
            plt.legend(loc = 'best')
            plt.savefig(save_path + sp + '_pos.png', format = 'png')