import copy
import glob
import os
import time
import datetime
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from rollout_storage import RolloutStorage
from replay_storage import ReplayStorage
from utils import get_vec_normalize
from visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'a2c-sil', 'ppo', 'ppo-sil', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def setup_dirs(experiment_name, log_dir, save_dir):
    log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    eval_log_dir = args.log_dir + "_eval"
    os.makedirs(eval_log_dir,  exist_ok=True)

    save_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    return log_dir, eval_log_dir, save_dir


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    experiment_name = args.env_name + '-' + args.algo + '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_dir, eval_log_dir, save_dir = setup_dirs(experiment_name, args.log_dir, args.save_dir)

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes,
        args.gamma, log_dir, args.add_timestep, device, False, frame_skip=0)

    if args.load_path:
        actor_critic, _ob_rms = torch.load(args.load_path)
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.train()
            vec_norm.ob_rms = _ob_rms
        actor_critic.train()
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            beta=args.beta_dist,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo.startswith('a2c'):
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               lr_schedule=args.lr_schedule,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo.startswith('ppo'):
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                         lr_schedule=args.lr_schedule,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    if args.algo.endswith('sil'):
        agent = algo.SIL(
            agent,
            update_ratio=args.sil_update_ratio,
            epochs=args.sil_epochs,
            batch_size=args.sil_batch_size,
            beta=args.sil_beta,
            value_loss_coef=args.sil_value_loss_coef,
            entropy_coef=args.sil_entropy_coef)
        replay = ReplayStorage(
            10000,
            num_processes=args.num_processes,
            gamma=args.gamma,
            prio_alpha=args.sil_alpha,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
            device=device)
    else:
        replay = None

    action_high = torch.from_numpy(envs.action_space.high).to(device)
    action_low = torch.from_numpy(envs.action_space.low).to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    benchmark_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                # sample actions
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            if args.clip_action and isinstance(envs.action_space, gym.spaces.Box):
                clipped_action = action.clone()
                clipped_action = torch.max(
                    torch.min(clipped_action, action_high), action_low)
            else:
                clipped_action = action

            # act in environment and observe
            obs, reward, done, infos = envs.step(clipped_action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    if 'rb' in info['episode']:
                        benchmark_rewards.append(info['episode']['rb'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)
            if replay is not None:
                replay.insert(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    action,
                    reward,
                    done)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, j, replay)

        rollouts.after_update()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        train_eprew = np.mean(episode_rewards)
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} episodes: mean/med {:.1f}/{:.1f}, min/max reward {:.2f}/{:.2f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       train_eprew,
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss), end='')
            if len(benchmark_rewards):
                print(", benchmark {:.1f}/{:.1f}, {:.1f}/{:.1f}".format(
                    np.mean(benchmark_rewards),
                    np.median(benchmark_rewards),
                    np.min(benchmark_rewards),
                    np.max(benchmark_rewards)
                ), end='')
            print()

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                if args.clip_action and isinstance(envs.action_space, gym.spaces.Box):
                    clipped_action = torch.max(
                        torch.min(action, action_high), action_low)
                else:
                    clipped_action = action

                obs, reward, done, infos = eval_envs.step(clipped_action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            eval_eprew = np.mean(eval_episode_rewards)
            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards), eval_eprew))

        if len(episode_rewards) and j % args.save_interval == 0 and save_dir != "":
            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None)]

            ep_rewstr = ("%d" % train_eprew).replace("-", "n")
            save_filename = os.path.join(save_dir, './checkpoint-%d-%s.pt' % (j, ep_rewstr))

            torch.save(save_model, save_filename)

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass


if __name__ == "__main__":
    main()
