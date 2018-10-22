import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import numpy as np
import argparse
import os
import gym
import torch

from envs import VecPyTorch, make_vec_envs
from utils import get_render_func, get_vec_normalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from envs import VecNormalize, VecPyTorch
from my_prosthetics_env import MyProstheticsEnv, project_obs


class StopTheSim(Exception):
    pass


class ClientWrapper(MyProstheticsEnv):

    def __init__(self, client, token):
        super(ClientWrapper, self).__init__(
            visualize=False,
            integrator_accuracy=1e-4,
            difficulty=0,
            seed=42)
        self.client = client
        self._cached_observation = self.client.env_create(token, env_id="ProstheticsEnv")
        print(self._cached_observation)
        self.step_count = 0

    def step(self, action, project=True):
        print('Step: ', self.step_count, end='. ')
        obs, reward, done, info = self.client.env_step(action.tolist())
        if obs is not None and 'body_pos' in obs:
            print('Pelvis: ', obs['body_pos']['pelvis'])
        elif obs is None:
            print('Invalid obs.')
            return None, None, True, None
        self.step_count += 1
        proj_obs = project_obs(obs, self.project_mode, self.prosthetic)
        return proj_obs, reward, done, info

    def reset(self, project=True):
        print('Reset')
        if self._cached_observation is not None:
            print('Returning cached')
            obs = self._cached_observation
            self._cached_observation = None
        else:
            obs = self.client.env_reset()
        self.step_count = 0
        if obs is None:
            raise StopTheSim
        return project_obs(obs, self.project_mode, self.prosthetic)

    def close(self):
        return self.client.env_close()


# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store', required=True)
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-path', default='',
                    help='directory to save agent logs (default: ')
args = parser.parse_args()

remote_base = 'http://grader.crowdai.org:1729'  # Submission to Round-1
# remote_base = 'http://grader.crowdai.org:1730'  # Submission to Round-2
client = Client(remote_base)


def create_env():
    env = ClientWrapper(client=client, token=args.token)
    return env

env = DummyVecEnv([create_env])
env = VecNormalize(env, ret=False)
env = VecPyTorch(env, 'cpu')

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(args.load_path)
actor_critic.eval()

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

# Create environment

ref_env = ProstheticsEnv()

obs = env.reset()

# Run a single step
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
count = 0
num_steps = 0
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True)

    clipped_action = action
    if isinstance(ref_env.action_space, gym.spaces.Box):
       clipped_action = torch.max(torch.min(
           clipped_action, torch.from_numpy(ref_env.action_space.high)),
           torch.from_numpy(ref_env.action_space.low))

    try:
        obs, reward, done, info = env.step(clipped_action)
        num_steps += 1
        if done:
            print('Done after %d steps.' % num_steps)
            num_steps = 0
            count += 1
    except StopTheSim:
        print('Finishing.')
        break

client.submit()
