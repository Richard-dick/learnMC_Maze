# Data Format
This repository contains three data sets: MC_Maze. These data sets were collected by academic laboratories and released by the [Neural Latents Benchmark](https://neurallatents.github.io/) in the [Neurodata Without Borders](https://www.nwb.org/) format. For convenience, they have been preprocessed into .pickle files that are smaller, trialized, and only contain data relevant to this code package.

Each .pickle file contains a dictionary `Data` that can be loaded via:
```
import pickle
dataset = 'area2_bump'
with open('data/' + dataset + '.pickle','rb') as f:
    Data = pickle.load(f)
```

`Data` is a dictionary with several keys (e.g., `spikes`, `condition`, `pos`), each of which contains trialized data. `spikes` is a list whose length matches the number of trials in the data set. Each element of the list contains an N x T numpy array of spike counts binned at 1 ms resolution where N is the number of neurons and T is the number of 1 ms samples in the trial. Behavioral variable groups like `pos` are similarly stored as a list of M x T numpy arrays where M is the number of components that describe that behavioral variable group (e.g, M = 2 for position when there is an x- and y-component). Area2_Bump and MC_Maze contain a `condition` key that is an numpy array of condition IDs, one per trial.

## MC_Maze
This data set contains neural recordings from M1 and PMd while a monkey makes delayed center-out straight and curved reaches. The monkey was presented with a virtual maze with targets to reach toward. Virtual barriers were presented that the monkey had to avoid colliding with when reaching, which prompted curved reaches to avoid the barriers. On some trials three targets were presented, but only one target was reachable via the maze. There are 2,295 trials in the data set provided, each beginning 549 ms before movement onset and ending 450 ms after movement onset.

### Conditions
There are 108 conditions in this data set: 36 maze configurations with 3 variants per maze. The `condition` field contains a condition number from 1 to 108. Condition numbers 1-3 correspond to one maze configuration, condition numbers 4-6 correspond to a second maze configuration, etc. Within each maze configuration, the first condition number corresponds to a reach to a target with no barrier (straight reach), the second condition number corresponds to reaching to a target with a barrier (curved reach), and the third condition number corresponds to reaching to a target with a barrier (curved reach) in the presence of two unreachable distractor targets.

### Behavioral Variable Groups

`pos`: x- and y-position of the monkey's hand (mm)

`vel`: x- and y-velocity of the monkey's hand (mm/s)

### Attribution
This data was provided to the Neural Latents Benchmark by Matt Kaufman, Mark Churchland, and Krishna Shenoy at Stanford University. The full data set is available on [DANDI](https://dandiarchive.org/dandiset/000128) and more information about the data can be found in the journal article [Churchland et al. 2010](https://pubmed.ncbi.nlm.nih.gov/21040842/).

