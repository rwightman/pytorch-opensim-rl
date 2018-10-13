import torch
import torch.nn as nn
import math

from envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))



class _Schedule:
    def __init__(self, initial_value, gamma=0.1, last_epoch=-1):
        self.gamma = gamma
        self.initial_value = initial_value
        self.last_epoch = last_epoch
        self.step()

    def get(self):
        return self.initial_value

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        return self.get()


class StepSchedule(_Schedule):

    def __init__(self, initial_value, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        super(StepSchedule, self).__init__(initial_value, gamma, last_epoch)

    def get(self):
        return self.initial_value  * self.gamma ** (self.last_epoch // self.step_size)


class ExpSchedule(_Schedule):

    def __init__(self, initial_value, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        super(ExpSchedule, self).__init__(initial_value, gamma, last_epoch)

    def get(self):
        return self.initial_value  * self.gamma ** (self.last_epoch / self.step_size)


class NatExpSchedule(_Schedule):

    def __init__(self, initial_value, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        super(NatExpSchedule, self).__init__(initial_value, gamma, last_epoch)

    def get(self):
        return self.initial_value * math.exp(-self.gamma * (self.last_epoch / self.step_size))
