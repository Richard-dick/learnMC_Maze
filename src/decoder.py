import numpy as np
from numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are

from src.utils import bin_spikes, append_history, array2list, zero_order_hold
import warnings




################################ KALMAN FILTER #################################

class KalmanFilter(object):

    """
    Class for the Kalman filter decoder

    Hyperparameters
    ---------------
    Delta: number of time points to pool into a time bin
    
    lag: number of time samples to lag behavioral variables relative to spiking data
        This accounts for the physiological latency between when neurons become active
        and when that activity impacts behavior.
    
    steady_state: boolean determining whether the steady-state form of the Kalman filter should be used
        The steady-state Kalman filter is much faster for prediction, but takes
        a few samples to converge to the same output as the standard Kalman filter.
    
    """

    def __init__(self,HyperParams):
        self.Delta = HyperParams['Delta']
        self.lag = HyperParams['lag']
        self.bin_lag = int(np.round(self.lag / self.Delta))
        self.Ts = .001 # hardcodes sampling period at 1 ms
        self.dt = self.Delta * self.Ts
        self.steady_state = HyperParams['steady_state']

    def fit(self, S, Z):

        """
        Train Kalman filter.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Z: list of 4 x T numpy arrays, each of which contains behavioral data for position and velocity
            The first two rows correspond to x- and y-position
            The last two rows correspond to x- and y-velocity
            
        Parameters (Steady-state Kalman filter)
        ----------
        A: state-transition matrix (7 x 7 numpy array)
        C: observation matrix (N x 7 numpy array)
        K_inf: steady-state Kalman gain (7 x N numpy array)
        z0: initial state mean (7 x 1 numpy 1D array)
        
        Parameters (Standard Kalman filter)
        ----------
        A: state-transition matrix (7 x 7 numpy array)
        C: observation matrix (N x 7 numpy array)
        Q: state-transition noise covariance (7 x 7 numpy array)
        R: observation noise covariance (N x N numpy array)
        z0: initial state mean (7 x 1 numpy 1D array)
        P0: initial state covariance (7 x 7 numpy array)
            
        """

        # Unpack attributes.
        Delta = self.Delta
        lag = self.lag
        dt = self.dt
        
        # Shift data to account for lag.
        if self.lag > 0:
            S = [s[:,:-lag] for s in S]
            Z = [z[:,lag:] for z in Z]

        # Compute acceleration. We are assuming that the first two components of Z are something
        # akin to position and the second two components are the something akin to velocity.
        acc = [np.hstack((np.zeros((2,1)),np.diff(z[2:4,:],axis=1)/self.Ts)) for z in Z]
        Z = [np.vstack((z,a)) for z,a in zip(Z,acc)]
        
        # Add a row of ones to allow for constant offset.
        Z = [np.vstack((z,np.ones(z.shape[1]))) for z in Z]
        
        # Downsample kinematics.
        Z = [z[:,Delta-1::Delta] for z in Z]

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]
        
        # Convert lists (one entry per trial) to large arrays.
        Z_init = np.concatenate([np.reshape(z[:,0], (-1, 1)) for z in Z], axis=1)
        Z1 = np.concatenate([z[:,:-1] for z in Z], axis=1)
        Z2 = np.concatenate([z[:,1:] for z in Z], axis=1)
        Z = np.concatenate(Z, axis=1)
        S = np.concatenate(S, axis=1)
        
        # Fit state transition matrix.
        A = Z2 @ Z1.T @ inv(Z1 @ Z1.T)
        
        # Fit measurement matrix.
        C = S @ Z.T @ inv(Z @ Z.T)
        
        # Fit state noise covariance.
        T1 = Z1.shape[1]
        Q = ((Z2 - A @ Z1) @ (Z2 - A @ Z1).T) / T1
        
        # Fit measurement noise covariance.
        T2 = Z.shape[1]
        R = ((S - C @ Z) @ (S - C @ Z).T) / T2
        
        # Fit initial state
        z0 = np.mean(Z_init, axis=1)
        P0 = np.cov(Z_init, bias=True)
        
        # Store parameters appropriate for standard or steady-state Kalman filter.
        if self.steady_state:
            
            try:
                # Compute steady-state Kalman gain.
                Q = (Q + Q.T)/2 # ensures Q isn't slightly asymmetric due to floating point errors before running 'dare'
                R = (R + R.T)/2 # ensures R isn't slightly asymmetric due to floating point errors before running 'dare'
                P_inf = solve_discrete_are(A.T,C.T,Q,R)
                K_inf = P_inf @ C.T @ pinv(C @ P_inf @ C.T + R)

                # Store parameters for steady-state Kalman filter.
                self.A = A
                self.C = C
                self.K_inf = K_inf
                self.z0 = z0

            except np.linalg.LinAlgError:

                # The 'solve_discrete_are' function won't always succeed due
                # to numerical properties of the matrices that get fed into
                # it. When it fails, issue a warning to the user letting them
                # know we'll have to revert to the standard Kalman filter.
                warn_str = '''
                Discrete-time algebraic Riccati equation could not be solved using the learned parameters. 
                Reverting from steady-state Kalman filter back to standard Kalman filter.'''
                warnings.warn(warn_str)

                # Revert to standard Kalman filter.
                self.steady_state = False
                self.A = A
                self.C = C
                self.Q = Q
                self.R = R
                self.z0 = z0
                self.P0 = P0

        else:

            # Store parameters for standard Kalman filter.
            self.A = A
            self.C = C
            self.Q = Q
            self.R = R
            self.z0 = z0
            self.P0 = P0

    def predict(self, S):

        """
        Predict behavior with trained Kalman filter.

        Inputs
        ------
        S: list of N x T numpy arrays, each of which contains spiking data for N neurons over T times

        Outputs
        -------
        Z_hat: list of 4 x T numpy arrays, each of which contains decoded position and velocity
            The first two rows correspond to x- and y-position estimates
            The last two rows correspond to x- and y-velocity estimates
   
        """

        # Unpack attributes.
        Delta = self.Delta
        bin_lag = self.bin_lag
        A = self.A
        C = self.C
        z0 = self.z0
        if self.steady_state:
            K_inf = self.K_inf
        else:
            Q = self.Q
            R = self.R
            P0 = self.P0

        # Compute trial lengths.
        T = [s.shape[1] for s in S]

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]
        
        # Create estimates for each trial.
        n_trials = len(S)
        Z_hat = [None] * n_trials
        for tr in range(n_trials):
            
            # Initialize Z_hat for this trial.
            M = A.shape[0]
            n_observations = S[tr].shape[1]
            Z_hat[tr] = np.zeros((M,n_observations))
            
            # Perform first measurement update.
            if self.steady_state:
                Z_hat[tr][:,0] = z0 + K_inf @ (S[tr][:,0] - C @ z0)
            else:
                P = P0
                K = P @ C.T @ pinv(C @ P @ C.T + R)
                Z_hat[tr][:,0] = z0 + K @ (S[tr][:,0] - C @ z0)
                P -= K @ C @ P
            
            # Estimate iteratively.
            for t in range(n_observations-1):
                
                # Perform time update.
                Z_hat[tr][:,t+1] = A @ Z_hat[tr][:,t]
                if not self.steady_state:
                    P = A @ P @ A.T + Q
                
                # Perform measurement update.
                if self.steady_state:
                    Z_hat[tr][:,t+1] += K_inf @ (S[tr][:,t+1] - C @ Z_hat[tr][:,t+1])
                else:
                    K = P @ C.T @ pinv(C @ P @ C.T + R)
                    Z_hat[tr][:,t+1] += K @ (S[tr][:,t+1] - C @ Z_hat[tr][:,t+1])
                    P -= K @ C @ P

        # Remove acceleration and constant offset from estimates.
        Z_hat = [Z[:4,:] for Z in Z_hat]

        # Prepend with NaNs to account for lag in estimates.
        if bin_lag > 0:
            Z_hat = [np.hstack((np.full((Z.shape[0],bin_lag), np.nan), Z)) for Z in Z_hat]

        # Return estimate to original time scale.
        Z_hat = [zero_order_hold(Z,Delta) for Z in Z_hat]
        Z_hat = [np.hstack((np.full((Z.shape[0],Delta-1), np.nan), Z)) for Z in Z_hat]
        Z_hat = [z[:,:t] for z,t in zip(Z_hat, T)]
                
        return Z_hat
