import numpy as np
from src.decoder import FeedforwardNetwork, TargetFFN
from bayes_opt import BayesianOptimization
from src.utils import partition

def train(train_var, config):
    # Unpack hyperparameters directly from config.
    HyperParams = config['general'].copy()
    beh_key = list(config.keys())
    beh_key.remove('general')
    beh_key.remove('opt')
    HyperParams.update(config[beh_key[0]])
    
    if beh_key[0] == 'pos':
        model = FeedforwardNetwork(HyperParams)
        model.fit(train_var['spikes'], train_var['behavior'])
    else:
        model = TargetFFN(HyperParams)
        model.fit(train_var['spikes'], train_var['behavior'], train_var['success'])
    
    return model, HyperParams
    



def optimize_hyperparams(S_train, S_val, Z_train, Z_val, gen_hp, opt_config):

    """
    Learn a good set of hyperparameters to use with a particular
    neural decoding method by using Bayesian optimization.

    Inputs
    ------
    S_train: list (1 x number of training trials) of N x T numpy arrays
        Each element of the list contains spiking data for N neurons over T times

    S_val: list (1 x number of validation trials) of N x T numpy arrays
        Each element of the list contains spiking data for N neurons over T times

    Z_train: list (1 x number of training trials) of M x T numpy arrays
        Each element of the list contains behavioral data for M behavioral variables over T times

    Z_val: list (1 x number of validation trials) of M x T numpy arrays
        Each element of the list contains behavioral data for M behavioral variables over T times

    gen_hp: dictionary of hyperparameters that are set and won't be optimized

    opt_config: dictionary specifying details of Bayesian optimization
        This contains general Bayesian optimization settings and 
        ranges to search for hyperparameters that will be optimized.

    Outputs
    -------
    HyperParams: dictionary of good hyperparameters learned via Bayesian optimization
   
    """

    def evaluate_model(**kwargs):
        
        # Create dictionary of hyperparameters.
        HyperParams = construct_hyperparams(kwargs, gen_hp)

        # Train model.
        model = FeedforwardNetwork(HyperParams)
        model.fit(S_train, Z_train)
        
        # Make predictions on validation set.
        Z_val_hat = model.predict(S_val)
        
        # Return mean R2 across decoded variables.
        tau = HyperParams['Bin_Size']*(HyperParams['tau_prime']+1)-1
        return np.mean(model.evaluate(Z_val, Z_val_hat, skip_samples=tau, eval_bin_size=5))
    
    # Unpack optimization settings.
    init_points = opt_config['init_points']
    n_iter = opt_config['n_iter']
    kappa = opt_config['kappa']

    # Reformat opt_config to store parameter bounds as tuples.
    pbounds = opt_config.copy()
    pbounds.pop('init_points')
    pbounds.pop('n_iter')
    pbounds.pop('kappa')
    pbounds.pop('val_frac')
    pbounds = {k: tuple(v) for k, v in pbounds.items()}

    # Optimize hyperparameters using the training and validation sets.
    optimizer = BayesianOptimization(evaluate_model, pbounds, verbose=1)
    optimizer.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa)
    best_params = optimizer.max['params']
    HyperParams = construct_hyperparams(best_params, gen_hp)

    return HyperParams

def construct_hyperparams(optimized_hp, gen_hp):

    """
    Construct a unified hyperparameters dictionary.

    Inputs
    ------
    optimized_hp: dictionary of hyperparameters that were
        optimized (or are in the process of being optimized)

    gen_hp: dictionary of general hyperparameters that
        were not optimized

    Outputs
    -------
    HyperParams: dictionary of hyperparameters
   
    """

    # Initialize with general hyperparameters.
    HyperParams = gen_hp.copy()
    HyperParams['num_units'] = int(optimized_hp['num_units'])
    HyperParams['num_layers'] = int(optimized_hp['num_layers'])
    HyperParams['frac_dropout'] = float(optimized_hp['frac_dropout'])
    HyperParams['num_epochs'] = int(optimized_hp['num_epochs'])


    return HyperParams
