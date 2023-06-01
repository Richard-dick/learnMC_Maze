import numpy as np
from numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are
import tensorflow as tf
import tensorflow.python.keras.layers as layers
from src.utils import bin_spikes, append_history, array2list, zero_order_hold
import warnings

    


########################## FEEDFORWARD NEURAL NETWORK ##########################

class FeedforwardNetwork(object):

    def __init__(self,HyperParams):
        # number of time points to pool into a time bin
        self.Bin_Size = HyperParams['Bin_Size']
        # ? number of previous time bins (in addition to the current bin) to use for decoding
        self.tau_prime = HyperParams['tau_prime']
        # number of units per hidden layer
        self.num_units = HyperParams['num_units']
        # number of hidden layers
        self.num_layers = HyperParams['num_layers']
        # ? unit dropout rate
        self.frac_dropout = HyperParams['frac_dropout']
        # number of training epochs
        self.num_epochs = HyperParams['num_epochs']

    def fit(self, spikes, behavior):
        # Unpack attributes.
        Bin_Size = self.Bin_Size
        tau_prime = self.tau_prime
        num_units = self.num_units
        num_layers = self.num_layers
        frac_dropout = self.frac_dropout
        num_epochs = self.num_epochs
    
        # Bin spikes
        # * list of N x T numpy arrays, each of which contains spiking data for N neurons over T times
        # 将list中每个元素(182, 1000)在axis=1(time)维度bin操作, 分箱大小为Bin_Size
        binned_spikes = [bin_spikes(sp, Bin_Size) for sp in spikes]
        
        # Downsample kinematics to bin width.
        # * list of M x T numpy arrays, each of which contains behavioral data for M behavioral variables over T times
        behavior = [b[:,Bin_Size-1::Bin_Size] for b in behavior]

        # Reformat observations to include recent history.
        # ! 现在spike多了一个轴, 用来表示一系列previous的时间, 大小为tau_prime+1
        appended_binned_spikes = [append_history(bs, tau_prime) for bs in binned_spikes]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        # * 现在只需要后36时刻的spikes信息了
        X = [abS[:,tau_prime:,:] for abS in appended_binned_spikes]
        behavior = [z[:,tau_prime:] for z in behavior]

        # Concatenate X and behavior across trials (in time bin dimension) and rearrange dimensions.
        # 将X的list中的axis=1(times)给合到一起(变成1836*36), 然后再调整为第一个维度
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [1, 0, 2])
        behavior = np.concatenate(behavior, axis=1).T

        # Z-score 归一化
        self.X_mu = np.mean(X, axis=0)
        self.X_sigma = np.std(X, axis=0)
        X = (X - self.X_mu) / self.X_sigma

        # Zero-center outputs.
        self.Z_mu = np.mean(behavior, axis=0)
        behavior = behavior - self.Z_mu

        # Construct feedforward network model.
        net = tf.keras.Sequential(name='Feedforward_Network')
        net.add(layers.Flatten())
        for layer in range(num_layers): # hidden layers
            net.add(layers.Dense(num_units, activation='relu'))
            if frac_dropout!=0: net.add(layers.Dropout(frac_dropout))
        net.add(layers.Dense(behavior.shape[1], activation='linear')) # output layer
        net.compile(optimizer="Adam", loss="mse", metrics="mse")

        # Fit model.
        net.fit(X, behavior, epochs=num_epochs)
        self.net = net

    def predict(self, Spikes):

        """
        Predict behavior with trained feedforward neural network.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Outputs
        -------
        Z_hat: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times
   
        """

        # Unpack attributes.
        Bin_Size = self.Bin_Size
        tau_prime = self.tau_prime
        X_mu = self.X_mu
        X_sigma = self.X_sigma
        Z_mu = self.Z_mu
        net = self.net

        # Store each trial's length.
        T = [s.shape[1] for s in Spikes]

        # Bin spikes.
        binned_spikes = [bin_spikes(sp, Bin_Size) for sp in Spikes]

        # Store each trial's bin length.
        T_prime = [bs.shape[1] for bs in binned_spikes]

        # Reformat observations to include recent history.
        append_binned_spikes = [append_history(s, tau_prime) for s in binned_spikes]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in append_binned_spikes]

        # Concatenate X across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [1, 0, 2])

        # Z-score inputs.
        X = (X - X_mu) / X_sigma

        # Generate predictions.
        Z_hat = net.predict(X)

        # Add mean back to outputs.
        Z_hat += Z_mu

        # Split Z_hat back into trials and transpose kinematic arrays.
        Z_hat = array2list(Z_hat, np.array(T_prime)-tau_prime, axis=0)
        Z_hat = [Z.T for Z in Z_hat]

        # Add NaNs where predictions couldn't be made due to insufficient spiking history.
        Z_hat = [np.hstack((np.full((Z.shape[0],tau_prime), np.nan), Z)) for Z in Z_hat]

        # Return estimate to original time scale.
        Z_hat = [zero_order_hold(Z,Bin_Size) for Z in Z_hat]
        Z_hat = [np.hstack((np.full((Z.shape[0],Bin_Size-1), np.nan), Z)) for Z in Z_hat]
        Z_hat = [z[:,:t] for z,t in zip(Z_hat, T)]

        return Z_hat

######################### GATED RECURRENT UNIT NETWORK #########################

class GRU(object):

    """
    Class for the GRU decoder

    Hyperparameters
    ---------------
    Bin_Size: number of time points to pool into a time bin
    
    tau_prime: number of previous time bins (in addition to the current bin) to use for decoding
    
    num_units: number of units in the GRU layer

    frac_dropout: unit dropout rate

    num_epochs: number of training epochs
    
    """

    def __init__(self,HyperParams):
        self.Bin_Size = HyperParams['Bin_Size']
        self.tau_prime = HyperParams['tau_prime']
        self.num_units = HyperParams['num_units']
        self.frac_dropout = HyperParams['frac_dropout']
        self.num_epochs = HyperParams['num_epochs']

    def fit(self, S, Z):

        """
        Train GRU.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Z: list of M x T numpy arrays, each of which contains behavioral data for M behavioral variables over T times
            
        Parameters
        ----------
        net: Keras sequential neural network model
            
        """

        # Unpack attributes.
        Bin_Size = self.Bin_Size
        tau_prime = self.tau_prime
        num_units = self.num_units
        frac_dropout = self.frac_dropout
        num_epochs = self.num_epochs

        # Bin spikes.
        S = [bin_spikes(sp, Bin_Size) for sp in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Downsample kinematics to bin width.
        Z = [z[:,Bin_Size-1::Bin_Size] for z in Z]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]
        Z = [z[:,tau_prime:] for z in Z]

        # Concatenate X and Z across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [2, 0, 1])
        Z = np.concatenate(Z, axis=1).T

        # Z-score inputs.
        X_mu = np.mean(X, axis=0)
        X_sigma = np.std(X, axis=0)
        X = (X - X_mu) / X_sigma
        self.X_mu = X_mu
        self.X_sigma = X_sigma

        # Zero-center outputs.
        Z_mu = np.mean(Z, axis=0)
        Z = Z - Z_mu
        self.Z_mu = Z_mu

        # Construct GRU network model.
        net = tf.keras.Sequential(name='GRU_Network')
        net.add(layers.GRU(num_units, dropout=frac_dropout, recurrent_dropout=frac_dropout))
        if frac_dropout!=0: net.add(layers.Dropout(frac_dropout))
        net.add(layers.Dense(Z.shape[1], activation='linear'))
        net.compile(optimizer="RMSprop", loss="mse", metrics="mse")

        # Fit model.
        net.fit(X, Z, epochs=num_epochs)
        self.net = net

    def predict(self, S):

        """
        Predict behavior with trained GRU.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Outputs
        -------
        Z_hat: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times
   
        """

        # Unpack attributes.
        Bin_Size = self.Bin_Size
        tau_prime = self.tau_prime
        X_mu = self.X_mu
        X_sigma = self.X_sigma
        Z_mu = self.Z_mu
        net = self.net

        # Store each trial's length.
        T = [s.shape[1] for s in S]

        # Bin spikes.
        S = [bin_spikes(sp, Bin_Size) for sp in S]

        # Store each trial's bin length.
        T_prime = [s.shape[1] for s in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]

        # Concatenate X across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [2, 0, 1])

        # Z-score inputs.
        X = (X - X_mu) / X_sigma

        # Generate predictions.
        Z_hat = net.predict(X)

        # Add mean back to outputs.
        Z_hat += Z_mu

        # Split Z_hat back into trials and transpose kinematic arrays.
        Z_hat = array2list(Z_hat, np.array(T_prime)-tau_prime, axis=0)
        Z_hat = [Z.T for Z in Z_hat]

        # Add NaNs where predictions couldn't be made due to insufficient spiking history.
        Z_hat = [np.hstack((np.full((Z.shape[0],tau_prime), np.nan), Z)) for Z in Z_hat]

        # Return estimate to original time scale.
        Z_hat = [zero_order_hold(Z,Bin_Size) for Z in Z_hat]
        Z_hat = [np.hstack((np.full((Z.shape[0],Bin_Size-1), np.nan), Z)) for Z in Z_hat]
        Z_hat = [z[:,:t] for z,t in zip(Z_hat, T)]

        return Z_hat
