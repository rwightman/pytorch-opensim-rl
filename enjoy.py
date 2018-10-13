import argparse
import os
import gym
import numpy as np
import torch

from envs import VecPyTorch, make_vec_envs
from utils import get_render_func, get_vec_normalize


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-path', default='',
                    help='directory to save agent logs (default: ')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--clip-action', action='store_true', default=False,
                    help='clip actions')
args = parser.parse_args()

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                    None, None, args.add_timestep, device='cpu',
                    allow_early_resets=False,
                    visualize=True)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(args.load_path)
actor_critic.eval()

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True)

    if args.clip_action and isinstance(env.action_space, gym.spaces.Box):
        clipped_action = torch.max(torch.min(
            action, torch.from_numpy(env.action_space.high)),
            torch.from_numpy(env.action_space.low))
    else:
        clipped_action = action

    # Obser reward and next obs
    obs, reward, done, _ = env.step(clipped_action)

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
