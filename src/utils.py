import numpy as np
from sklearn.model_selection import train_test_split

def partition(condition:list, div_frac):
    # Number the trials.
    trial_idx = np.arange(len(condition))
    
    # Partition the data differently depending on whether there is condition structure.
    n_conds = len(np.unique(np.array(condition)))
    n_test = int(div_frac*len(condition))
    if n_test >= n_conds:
        train_idx, val_idx = train_test_split(trial_idx, test_size=div_frac, stratify=condition)
    else:
        train_idx, val_idx = train_test_split(trial_idx, test_size=div_frac)

    return train_idx, val_idx

################################################################################

def bin_spikes(spikes:np.array, bin_size:np.array) -> np.array:
    # Get some useful constants.
    [neuron_samples, time_samples] = spikes.shape
    bins = int(time_samples/bin_size) # number of time bins

    # Count spikes in bins.
    binned_spikes = np.empty([neuron_samples, bins])
    for idx in range(bins):
        binned_spikes[:, idx] = np.sum(spikes[:, idx*bin_size:(idx+1)*bin_size], axis=1)

    return binned_spikes

################################################################################

def bin_kin(Z, bin_size):

    """
    Bin behavioral variables in time.

    Inputs
    ------
    Z: numpy array of behavioral variables (behaviors x time)

    bin_size: number of time points to pool into a time bin

    Outputs
    -------
    Z_bin: numpy array of binned behavioral variables (behaviors x bins)
   
    """

    # Get some useful constants.
    [M, n_time_samples, tau] = Z.shape
    K = int(n_time_samples/bin_size) # number of time bins

    # Average kinematics within bins.
    Z_bin = np.empty([M, K, tau])
    for k in range(K):
        Z_bin[:, k,:] = np.mean(Z[:, k*bin_size:(k+1)*bin_size,:], axis=1)

    return Z_bin

################################################################################

def append_history(Spikes:np.ndarray, tau_prime:int) -> np.ndarray:
    # Get some useful constants.
    [N, K] = Spikes.shape # [number of neurons, number of bins]

    S_aug = np.empty([N, K-tau_prime, tau_prime])
    for i in range(K-tau_prime):
        S_aug[:, i, :] = Spikes[:, i:i+tau_prime]
    
    return S_aug

################################################################################

def array2list(array, sizes, axis):
    
    # Get indices indicating where to divide array.
    split_idx = np.cumsum(sizes)
    split_idx = split_idx[:-1]
    
    # Split up array.
    array_of_arrays = np.split(array, split_idx, axis=axis)
    
    # Convert outer array to list.
    list_of_arrays = list(array_of_arrays)

    return list_of_arrays

################################################################################

def zero_order_hold(binned_data, bin_size):

    return np.repeat(binned_data, bin_size, axis=1)

################################################################################

def pad_to_length(data, T):

    """
    Pad data with final value as needed to reach a specified length.

    Inputs
    ------
    data: numpy array (variables x time)

    T: number of desired time samples

    Outputs
    -------
    padded_data: numpy array (variables x T)
   
    """

    # Initialized padded_data as data.
    padded_data = data

    # If padding is necessary...
    n_samples = data.shape[1]
    if n_samples < T:
        final_value = data[:,-1]
        pad_len = T - n_samples
        pad = np.tile(final_value,(pad_len,1)).T
        padded_data = np.hstack((padded_data,pad))

    return padded_data

################################################################################


