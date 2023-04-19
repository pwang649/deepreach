import configargparse
import inspect
import os
import torch
import random
import numpy as np
import pickle

from datetime import datetime
from dynamics import dynamics
import experiments
import modules
import dataio

p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# save/load directory options
p.add_argument('--experiments_dir', type=str, default='./runs', help='Where to save the experiment subdirectory.')
p.add_argument('--experiment_name', type=str, required=True, help='Name of the experient subdirectory.')

p.add_argument('--dt', type=float, default=0.0025, help='The dt used in testing simulations')
p.add_argument('--checkpoint_toload', type=int, default=None, help="The checkpoint to load for testing (-1 for final training checkpoint, None for cross-checkpoint testing")
p.add_argument('--num_scenarios', type=int, default=100000, help='The number of scenarios sampled in scenario optimization for testing')
p.add_argument('--num_violations', type=int, default=1000, help='The number of violations to sample for in scenario optimization for testing')
p.add_argument('--control_type', type=str, default='value', choices=['value', 'ttr', 'init_ttr'], help='The controller to use in scenario optimization for testing')
p.add_argument('--data_step', type=str, default='run_basic_recovery', choices=['plot_violations', 'run_basic_recovery', 'plot_basic_recovery', 'collect_samples', 'train_binner', 'run_binned_recovery', 'plot_binned_recovery', 'plot_cost_function'], help='The data processing step to run')

opt = p.parse_args()

experiment_dir = os.path.join(opt.experiments_dir, opt.experiment_name)

# confirm that experiment dir already exists
if not os.path.exists(experiment_dir):
    raise RuntimeError('Cannot run test mode: experiment directory not found!')

current_time = datetime.now()

# load original experiment settings
with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'rb') as opt_file:
    orig_opt = pickle.load(opt_file)

# set the experiment seed
torch.manual_seed(orig_opt.seed)
random.seed(orig_opt.seed)
np.random.seed(orig_opt.seed)

dynamics_class = getattr(dynamics, orig_opt.dynamics_class)
dynamics = dynamics_class(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'})

dataset = dataio.ReachabilityDataset(
    dynamics=dynamics, numpoints=orig_opt.numpoints,
    pretrain=orig_opt.pretrain, pretrain_iters=orig_opt.pretrain_iters,
    tMin=orig_opt.tMin, tMax=orig_opt.tMax,
    counter_start=orig_opt.counter_start, counter_end=orig_opt.counter_end,
    num_src_samples=orig_opt.num_src_samples)

model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                             final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
model.cuda()

experiment_class = getattr(experiments, orig_opt.experiment_class)
experiment = experiment_class(model=model, dataset=dataset, experiment_dir=experiment_dir)
experiment.init_special(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(experiment_class.init_special).parameters.keys() if argname != 'self'})

experiment.test(
    current_time=current_time,
    last_checkpoint=orig_opt.num_epochs, checkpoint_dt=orig_opt.epochs_til_ckpt,
    checkpoint_toload=opt.checkpoint_toload, dt=opt.dt,
    num_scenarios=opt.num_scenarios, num_violations=opt.num_violations,
    set_type='BRT' if orig_opt.minWith in ['zero', 'target'] else 'BRS', control_type=opt.control_type, data_step=opt.data_step)
