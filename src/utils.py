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

def append_history(Spikes, tau_prime):
    # Get some useful constants.
    [N, K] = Spikes.shape # [number of neurons, number of bins]

    # Augment matrix with recent history.
    S_aug = np.empty([N, K, tau_prime+1])
    for i in range(-tau_prime,0):
        # 单论每个时刻, 左边是|i|个nan, 右边是从 size 片选到-i的spikes
        S_aug[:, :, i+tau_prime] = np.hstack((np.full([N,-i], np.nan), Spikes[:, :i]))
    S_aug[:, :, tau_prime] = Spikes

    return S_aug

################################################################################

def array2list(array, sizes, axis):

    """
    Break up a numpy array along a particular axis into a list of arrays.

    Inputs
    ------
    array: numpy array to break up

    sizes: vector of sizes for the resulting arrays along the specified axis (should sum to input array size along this axis)

    axis: axis to break up array along

    Outputs
    -------
    list_of_arrays: list where each element is a numpy array
   
    """
    
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

    """
    Upsample data with zero-order hold.

    Inputs
    ------
    binned_data: numpy array (variables x bins)

    bin_size: number of samples that were pooled into each time bin

    Outputs
    -------
    unbinned_data: numpy array of zero-order-held data (variables x time)
   
    """
    
    # Preallocate unbinned data.
    # n_vars, n_bins, n_taus = binned_data.shape
    # unbinned_data = np.zeros((n_vars, n_bins*bin_size, n_taus))
    
    # Upsample data with zero-order hold.
    # for k in range(n_bins):
        # print(binned_data.shape)
        # tmp = np.tile(binned_data[:,k,:], bin_size)
        # print(tmp.shape)
        # unbinned_data[:,k*bin_size:(k+1)*bin_size,:] = np.repeat(binned_data, 16, axis=1)
        # exit(0)

    return np.repeat(binned_data, 16, axis=1)

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

def evaluate(Z, Z_hat, skip_samples=0, eval_bin_size=1):

    """
    Compute the coefficients of determination (R2).

    Inputs
    ------
    Z: list of ground truth matrices (behaviors x observations)

    Z_hat: list of predicted values matrices (behaviors x observations)

    skip_samples: number of observations to exclude at the start of each trial
        Methods that generate predictions causally may not have predictions
        for the first few bins of a trial. This input allows these bins to
        be excluded from the R2 computation.

    eval_bin_size: number of observations to bin in time (by averaging) before computing R2

    Outputs
    -------
    R2: numpy array of R2s (one per behavioral variable)

    """

    # Remove some samples at the beginning of each
    # trial that were flagged to be skipped.
    # print(Z_hat.shape)
    # print(Z.shape)
    Z = [append_history(beh, 12) for beh in Z]
    Z = [z[:,skip_samples:,:] for z in Z]
    
    Z_hat = [z[:,skip_samples:,:] for z in Z_hat]

    # Bin kinematics in time.
    Z = [bin_kin(z, eval_bin_size) for z in Z]
    Z_hat = [bin_kin(z, eval_bin_size) for z in Z_hat]

    # Concatenate lists.
    Z = np.concatenate(Z,1)
    Z_hat = np.concatenate(Z_hat,1)
    # print(Z.shape)

    # Compute residual sum of squares.
    SS_res = np.sum((Z - Z_hat)**2, axis=1)
    # print(SS_res)
    
    Z_mu = [np.mean(Z, axis=1)] * Z.shape[1]

    # Compute total sum of squares.
    Z_mu = np.transpose(Z_mu,[1,0,2])
    SS_tot = np.sum((Z - Z_mu)**2, axis=1)

    # Compute coefficient of determination.
    R2 = 1 - SS_res/SS_tot

    return R2
