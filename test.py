## import package
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pickle
import pandas as pd
import os



## using my func
print("start reading")

dataset = NWBDataset("./data/000128/sub-Jenkins/", "*test", split_heldout=False)     

if not os.path.exists('data/pickle'):
    os.makedirs('data/pickle')
print("reading over, now transferring")
trial_info = dataset.trial_info
trial_data = dataset.data
# * 开始转成需要的格式
## condition
print(trial_data.keys())
exit(0)
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


with open('data/pickle/mc_maze_test.pickle', 'wb') as f:
    pickle.dump(data, f)
    