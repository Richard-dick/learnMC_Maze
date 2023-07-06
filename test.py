## Imports
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pandas as pd


print("start reading")

dataset = NWBDataset("./data/000128/sub-Jenkins/", "*train", split_heldout=False)     
print("reading over, now transferring")
trial_info = dataset.trial_info

reaction_time = trial_info["rt"]
print("reaction_time")
print(reaction_time.mean())
print(reaction_time.median())
delayed_time = trial_info["delay"]
print("delayed_time")
print(delayed_time.mean())
print(delayed_time.median())
