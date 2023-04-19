import math

import torch
import os
import shutil
import numpy as np

from abc import ABC, abstractmethod
from error_evaluators import scenario_optimization, ValueThresholdValidator, target_fraction, SliceSampleGenerator


class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):

        self.model.eval()
        self.model.requires_grad_(False)

        testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        if os.path.exists(testing_dir):
            overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
            if not (overwrite == 'y'):
                print('Exiting.')
                quit()
            shutil.rmtree(testing_dir)
        os.makedirs(testing_dir)

        self._load_checkpoint(checkpoint_toload)

        model = self.model
        dataset = self.dataset
        dynamics = dataset.dynamics

        # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
        beta = 1e-16
        epsilon = 1e-3
        N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))

        # 1. execute algorithm for tMax
        # record state/learned_value/violation for each while loop iteration
        delta_level = float('inf') if dynamics.set_mode == 'reach' else float('-inf')
        algorithm_iters = []

        results = scenario_optimization(
            model=model, dynamics=dynamics,
            tMin=dataset.tMin, t=dataset.tMax, dt=dt,
            set_type=set_type, control_type=control_type,
            scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
            sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
            sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
            violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            max_scenarios=N, max_samples=1000*min(N, 10000))

        algorithm_iters.append(
            {
                'states': results['states'],
                'values': results['values'],
                'violations': results['violations']
            }
        )
        violation_levels = results['values'][results['violations']]
        delta_level_arg = np.argmin(violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
        delta_level = violation_levels[delta_level_arg].item()
        print('violation_rate:', str(results['violation_rate']))
        print('delta_level:', str(delta_level))
        print('valid_sample_fraction:', str(results['valid_sample_fraction'].item()))


class VerifyDeepReach(Experiment):
    def init_special(self):
        pass
