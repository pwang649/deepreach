from abc import ABC, abstractmethod
import diff_operators

import math
import torch

# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
class Dynamics(ABC):
    def __init__(self, state_dim:int, input_dim:int, control_dim:int, disturbance_dim:int, state_mean:list, state_var:list, value_mean:float, value_var:float, value_normto:float, diff_model:bool):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean)
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.diff_model = diff_model
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + str(len(state_descriptor)) + ' != ' + str(self.state_dim)

    # ALL METHODS ARE BATCH COMPATIBLE

    # MODEL-UNIT CONVERSIONS (TODO: refactor into separate model-unit conversion class?)

    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.diff_model:
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.diff_model:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]

        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError

class Air3D(Dynamics):
    def __init__(self, collisionR:float, velocity:float, omega_max:float, angle_alpha_factor:float):
        self.collisionR = collisionR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        super().__init__(
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=1,
            state_mean=[0, 0, 0],
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            diff_model=False,
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # Air3D dynamics
    # \dot x    = -v + v \cos \psi + u y
    # \dot y    = v \sin \psi - u x
    # \dot \psi = d - u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = -self.velocity + self.velocity*torch.cos(state[..., 2]) + control[..., 0]*state[..., 1]
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2]) - control[..., 0]*state[..., 0]
        dsdt[..., 2] = disturbance[..., 0] - control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.collisionR

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        ham = self.omega_max * torch.abs(dvds[..., 0] * state[..., 1] - dvds[..., 1] * state[..., 0] - dvds[..., 2])  # Control component
        ham = ham - self.omega_max * torch.abs(dvds[..., 2])  # Disturbance component
        ham = ham + (self.velocity * (torch.cos(state[..., 2]) - 1.0) * dvds[..., 0]) + (self.velocity * torch.sin(state[..., 2]) * dvds[..., 1])  # Constant component
        return ham

    def optimal_control(self, state, dvds):
        det = dvds[..., 0]*state[..., 1] - dvds[..., 1]*state[..., 0]-dvds[..., 2]
        return (self.omega_max * torch.sign(det))[..., None]

    def optimal_disturbance(self, state, dvds):
        return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', 'theta'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class MultiVehicleCollision(Dynamics):
    def __init__(self, diff_model):
        self.set_mode = 'avoid'
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        super().__init__(
            state_dim=9, input_dim=10, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0,
                0, 0,
                0, 0, 0,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi,
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            diff_model=diff_model
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 8] = (wrapped_state[..., 8] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        return dsdt

    def boundary_fn(self, state):
        boundary_values = torch.norm(state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] - state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    def optimal_control(self, state, dvds):
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                0, 0,
                -0.4, 0,
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }
