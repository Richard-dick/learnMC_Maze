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
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

## using my func
from src.format import picklize, load_data
PICKLIZED = True



if __name__ == '__main__':
    if not PICKLIZED:
        picklize("MC_Maze_S")
    train_data, test_data = load_data("MC_Maze_S")
    print(type(train_data))
    print(type(test_data))
        
    

exit(0)



# print(type(Dataset.data))

# print(Dataset.trial_info["barrier_pos"])

# desc = Dataset.descriptions
# data = Dataset.data

# # for key in desc.keys():
# #     print(key+":")
# #     print("description:" + desc[key])
# #     print(data[str(key)])

# print(data["cursor_pos"])

## Plot trial-averaged reaches

# Find unique conditions
conds = dataset.trial_info.set_index(['trial_type', 'trial_version']).index.unique().tolist()


# Initialize plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Loop over conditions and compute average trajectory
for cond in conds:
    # Find trials in condition
    mask = np.all(dataset.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
    print(cond)
    print(mask)
    # Extract trial data
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-50, 450), ignored_trials=(~mask))
    # Average hand position across trials
    traj = trial_data.groupby('align_time')[[('hand_pos', 'x'), ('hand_pos', 'y')]].mean().to_numpy()
    # Determine reach angle for color
    active_target = dataset.trial_info[mask].target_pos.iloc[0][dataset.trial_info[mask].active_target.iloc[0]]
    reach_angle = np.arctan2(*active_target[::-1])
    # Plot reach
    ax.plot(traj[:, 0], traj[:, 1], linewidth=0.7, color=plt.cm.hsv(reach_angle / (2*np.pi) + 0.5))
    # exit(0)

plt.axis('off')
plt.savefig("test.png")
