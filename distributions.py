import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AddBias, init, init_normc_

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)
FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
entropy_normal = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy_normal(self).sum(-1)
FixedNormal.mode = lambda self: self.mean

FixedBeta = torch.distributions.Beta
entropy_beta = FixedBeta.entropy
FixedBeta.entropy = lambda self: entropy_beta(self).sum(-1)
FixedBeta.mode = lambda self: (self.concentration1 - 1) / (self.concentration1 + self.concentration0 - 2)
log_prob_beta = FixedBeta.log_prob
FixedBeta.log_probs = lambda self, actions: log_prob_beta(self, actions).sum(-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Beta(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Beta, self).__init__()

        init_ = lambda m: init(
            m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.fc_a = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_b = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        action_a = self.fc_a(x)
        action_b = self.fc_b(x)
        action_a = F.softplus(action_a) + 1.
        action_b = F.softplus(action_b) + 1.

        return FixedBeta(action_a, action_b)

