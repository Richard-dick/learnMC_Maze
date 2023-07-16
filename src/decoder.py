import numpy as np
from numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are
import tensorflow as tf
import tensorflow.python.keras.layers as layers
from src.utils import bin_spikes, append_history, array2list, zero_order_hold, bin_kin
import matplotlib.pyplot as plt
import warnings
import os



########################## FEEDFORWARD NEURAL NETWORK ##########################

class FeedforwardNetwork(object):

    def __init__(self,HyperParams):
        # number of time points to pool into a time bin
        self.Bin_Size = HyperParams['Bin_Size']
        self.spikes_tau_prime = HyperParams['spikes_tau_prime']
        self.behavior_tau_prime = HyperParams['behavior_tau_prime']
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
        spikes_tau_prime = self.spikes_tau_prime
        behavior_tau_prime = self.behavior_tau_prime
        num_units = self.num_units
        num_layers = self.num_layers
        frac_dropout = self.frac_dropout
        num_epochs = self.num_epochs
    
        # * list of N x T numpy arrays, each of which contains spiking data for N neurons over T times
        # 将list中每个元素(182, 1024)在axis=1(time)维度 bin 操作, 分箱大小为 16
        # 为(182, 64)
        X = [bin_spikes(sp, Bin_Size) for sp in spikes]
        
        # Downsample kinematics to bin width.
        # * list of M x T numpy arrays, each of which contains behavioral data for M behavioral variables over T times
        behavior = [b[:,Bin_Size-1::Bin_Size] for b in behavior]

        # Reformat observations to include recent history.
        # ! 现在spike多了一个轴, 用来表示一系列previous的时间, 大小为tau_prime+1
        appended_spikes = [append_history(bs, spikes_tau_prime) for bs in X]
        # print(appended_spikes[0].shape)
        # exit(0)

        behavior = [append_history(beh, behavior_tau_prime) for beh in behavior]
        # print(behavior[0].shape)
        # exit(0)
        appended_spikes = [ bs[:,:-behavior_tau_prime,:] for bs in appended_spikes]
        behavior = [ beh[:,spikes_tau_prime:,:] for beh in behavior]

        # Concatenate X and behavior across trials (in time bin dimension) and rearrange dimensions.
        # 将X的list中的axis=1(times)给合到一起, 然后再调整为第一个维度
        X = np.moveaxis(np.concatenate(appended_spikes,axis=1), [0, 1, 2], [1, 0, 2])
        behavior = np.moveaxis(np.concatenate(behavior,axis=1), [0, 1, 2], [1, 0, 2])
        # print(X.shape)
        # print(behavior.shape)
        # exit(0)
        
        # Z-score 归一化
        self.X_mu = np.mean(X, axis=0)
        # self.X_sigma = np.std(X, axis=0) + 0.000001
        # print(self.X_sigma.shape)
        X = (X - self.X_mu) 

        # Zero-center outputs.
        self.Z_mu = np.mean(behavior, axis=0)
        behavior = behavior - self.Z_mu
        
        out_shape = [behavior.shape[1], behavior.shape[2]]

        # Construct feedforward network model.
        net = tf.keras.Sequential(name='Feedforward_Network')
        net.add(layers.Flatten())
        for layer in range(num_layers): # hidden layers
            net.add(layers.Dense(num_units, activation='relu'))
            if frac_dropout!=0: net.add(layers.Dropout(frac_dropout))
        # net.add(layers.Dense(out_shape, activation='linear')) # output layer
        net.add(layers.Dense(out_shape[0]*out_shape[1], activation='linear')) 
        # 重塑输出张量的形状为 [1836, 2, 13]
        net.add(layers.Reshape((out_shape[0], out_shape[1]))) 
        net.compile(optimizer="Adam", loss="mse", metrics="mse")

        # Fit model.
        net.fit(X, behavior, epochs=num_epochs)
        self.net = net

    def predict(self, Spikes):

        # Unpack attributes.
        Bin_Size = self.Bin_Size
        spikes_tau_prime = self.spikes_tau_prime
        behavior_tau_prime = self.behavior_tau_prime
        X_mu = self.X_mu
        Z_mu = self.Z_mu
        net = self.net

        # Bin spikes.
        X = [bin_spikes(sp, Bin_Size) for sp in Spikes]

        # Store each trial's bin length.
        T_prime = [bs.shape[1] for bs in X]
        # print(T_prime)

        # Reformat observations to include recent history.
        append_binned_spikes = [append_history(s, spikes_tau_prime) for s in X]
        appended_spikes = [ bs[:,:-behavior_tau_prime,:] for bs in append_binned_spikes]
        # Concatenate X across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(appended_spikes, axis=1), [0, 1, 2], [1, 0, 2])

        # Z-score inputs.
        X = (X - X_mu) 

        # Generate predictions.
        # print(X.shape)
        Z_hat = net.predict(X)
        
        # Add mean back to outputs.
        Z_hat += Z_mu

        # Split Z_hat back into trials and transpose kinematic arrays.
        Z_hat = array2list(Z_hat, np.array(T_prime)-spikes_tau_prime-behavior_tau_prime, axis=0)
        
        Z_hat = [np.transpose(Z, (1, 0, 2)) for Z in Z_hat]
        # print(len(Z_hat))
        # print(Z_hat[0].shape)
        # exit(0)

        return Z_hat
    
    def evaluate(self, Z, Z_hat, visulize = False, save_dir = ""):

        Bin_Size = self.Bin_Size
        spikes_tau_prime = self.spikes_tau_prime
        behavior_tau_prime = self.behavior_tau_prime
        
        vis_item = np.random.randint(0, len(Z))    
        x_ref,y_ref = Z[vis_item]
        
        Z = [b[:,Bin_Size-1::Bin_Size] for b in Z]
        Z = [append_history(beh, behavior_tau_prime) for beh in Z]
        Z = [ beh[:,spikes_tau_prime:,:] for beh in Z]
        
        # print(Z[0].shape)
        # print(Z_hat[0].shape)
        # exit(0)

        # 选取一个实验结果的24条轨迹, 画出来
        if visulize:
            
            print("starting visualizing :"+ str(vis_item))
            save_path = "results/" + save_dir + '/trace/'
            if os.path.exists(save_path):
                import shutil
                shutil.rmtree(save_path)
            os.makedirs(save_path)
            z, z_hat = Z[vis_item], Z_hat[vis_item] 
            for i in range(z_hat.shape[1]):
                x,y = z[0,i,:],z[1,i,:]
                x_hat,y_hat = z_hat[0,i,:],z_hat[1,i,:]
                plt.cla()
                plt.plot(x_ref, y_ref, "y-", label = "reference")
                plt.plot(x, y, "r-", label = "ref_trace" )
                plt.plot(x_hat,y_hat, "g-", label = "res_trace")
                plt.title('The '+str(i)+' trace')
                
                plt.xlabel('x(mm)')
                plt.ylabel('y(mm)')
                plt.legend(loc = "best")
                
                plt.savefig(save_path+str(i)+'.png', format = 'png')
        
        mse = 0
        for i in range(len(Z)):
            error = Z[i] - Z_hat[i]
            mse_per_round = np.mean(np.square(error), axis=(0,2))
            mse = mse + mse_per_round
        mse = mse / len(Z)
            

        return mse

########################## FEEDFORWARD NEURAL NETWORK FOR AIM_POS ##########################

class TargetFFN(object):

    def __init__(self,HyperParams):
        # number of time points to pool into a time bin
        self.Bin_Size = HyperParams['Bin_Size']
        self.spikes_tau_prime = HyperParams['spikes_tau_prime']
        self.behavior_tau_prime = HyperParams['behavior_tau_prime']
        # number of units per hidden layer
        self.num_units = HyperParams['num_units']
        # number of hidden layers
        self.num_layers = HyperParams['num_layers']
        # ? unit dropout rate
        self.frac_dropout = HyperParams['frac_dropout']
        # number of training epochs
        self.num_epochs = HyperParams['num_epochs']

    def fit(self, spikes, behavior, success):
        # Unpack attributes.
        Bin_Size = self.Bin_Size
        num_units = self.num_units
        num_layers = self.num_layers
        frac_dropout = self.frac_dropout
        num_epochs = self.num_epochs
        
        # * 删除未成功的实验
        true_idx = [i for i in range(len(success)) if success[i] is True]
        X = [spikes[i] for i in true_idx]
        # * list of target_pos, each of which contains the last pos in one trial
        Z = [behavior[i] for i in true_idx]
    
        # * list of N x T numpy arrays, each of which contains spiking data for N neurons over T times
        # 将list中每个元素(??, 1024)在axis=1(time)维度 bin 操作, 分箱大小为 16
        # 为(??, 64)
        X = [bin_spikes(sp, Bin_Size) for sp in X]
        plan_time = X[0].shape[1] // 2
        # 然后取出前半段时间
        X = [sp[:, :plan_time] for sp in X]

        # Concatenate X and behavior across trials (in time bin dimension) and rearrange dimensions.
        # 将X的list中的axis=1(times)给合到一起, 然后再调整为第一个维度
        # X = np.moveaxis(np.concatenate(X,axis=1), [0, 1], [1, 0])
        X = np.array(X)
        Z = np.array(Z)
        
        # Z-score 归一化
        self.X_mu = np.mean(X, axis=0)
        X = (X - self.X_mu) 

        # Zero-center outputs.
        self.Z_mu = np.mean(behavior, axis=0)
        behavior = Z - self.Z_mu

        # Construct feedforward network model.
        net = tf.keras.Sequential(name='Feedforward_Network')
        net.add(layers.Flatten())
        for layer in range(num_layers): # hidden layers
            net.add(layers.Dense(num_units, activation='relu'))
            if frac_dropout!=0: net.add(layers.Dropout(frac_dropout))
        net.add(layers.Dense(behavior.shape[1], activation='linear')) 
        net.compile(optimizer="Adam", loss="mse", metrics="mse")

        # Fit model.
        net.fit(X, behavior, epochs=num_epochs)
        self.net = net

    def predict(self, Spikes):
        # Unpack attributes.
        Bin_Size = self.Bin_Size
        X_mu = self.X_mu
        Z_mu = self.Z_mu
        net = self.net
    
        # * 处理 X
        X = [bin_spikes(sp, Bin_Size) for sp in Spikes]
        plan_time = X[0].shape[1] // 2
        # 然后取出前半段时间
        X = [sp[:, :plan_time] for sp in X]

        # Concatenate X and behavior across trials (in time bin dimension) and rearrange dimensions.
        # 将X的list中的axis=1(times)给合到一起, 然后再调整为第一个维度
        # X = np.moveaxis(np.concatenate(X,axis=1), [0, 1], [1, 0])
        X = np.array(X)
        
        # Z-score inputs.
        X = (X - X_mu) 

        # Generate predictions.
        Z_hat = net.predict(X)
        
        # Add mean back to outputs.
        Z_hat += Z_mu

        # print(len(Z_hat))
        # print(Z_hat.shape)
        # exit(0)

        return Z_hat
    
    def evaluate(self, Z, Z_hat, visulize = False, save_dir = ""):
        
        Z = np.array(Z)
        
        if visulize:
            print("starting visualizing the target_pos")
            save_path = "results/" + save_dir + '/pos/'
            if os.path.exists(save_path):
                import shutil
                shutil.rmtree(save_path)
            os.makedirs(save_path)

            # 提取 x 和 y 坐标
            x = Z[:, 0]
            y = Z[:, 1]
            x_hat = Z_hat[:, 0]
            y_hat = Z_hat[:, 1]

            # 绘制散点图
            plt.scatter(x, y, color = 'green', label = "ref_pos")
            plt.scatter(x_hat, y_hat, color = 'red', label = "res_pos")
            plt.xlabel('x(mm)')
            plt.ylabel('y(mm)')
            plt.title('Scatter Plot of Pos')
            plt.legend(loc = 'best')
            plt.savefig(save_path +'pos.png', format = 'png')
        
        error = Z - Z_hat
        mse = np.mean(np.square(error), axis=0)           

        return mse