# PyTorch Reinforcement Learning for OpenSim Environments

This is my code for experimenting with the CrowdAI Prosthetics Challenge (https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge)

The reinforcement learning codebase is based upon Ilya Kostrikov's awesome work (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

As this is part of my learning process for continuous control with deep reinforcement learning, there are likely to be some issues.

All experiments were performed with PPO or PPO w/ self-improvement learning w/ 16 vector'd environments running in parallel. Keep in mind, the simulator is VERY slow so expect to wait a long time for decent results (days) -- even if you happen to have a kick ass machine.

Added:
 * support for the OpenSim Gym-like environments with Ilya's RL codebase
 * custom 'MyProstheticsEnv' wrapper to allow easier experimentation with different observation projections, rewards, and other aspects
 * frame skipping support in custom env
 * beta distribution experiment for continuous control in the range [0, 1] (http://ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)
 * tweaks to logging/folders/checkpoints and model resume for easier experimentation and tracking of results
 * an implementation of SIL (https://arxiv.org/abs/1806.05635), one variant off policy replay with on policy methods. It speeds initial training but starts to falter. I need further experiments with loss weight and other sil param decay.
 
 
 ## Get Started
 
 Setup your environment as per https://github.com/stanfordnmbl/osim-rl#getting-started
 
 ## Give It a Go
  
 Unclipped -- trains much faster but not clear what OpenSim is doing:
 `main.py --algo ppo --env-name osim.Prosthetics --lr 7e-4 --num-steps 1000 --use-gae --ppo-epoch 10`
 
 With clipped [0, 1] actions shifted so mean is at 0.5:
 
 `main.py --algo ppo --env-name osim.Prosthetics --lr 1e-3 --num-steps 1000 --use-gae --ppo-epoch 10 --clip-action -shift-action`
 
 With beta distribution [0, 1]:
 
`main.py --algo ppo --env-name osim.Prosthetics --lr 1e-3 --num-steps 1000 --use-gae --ppo-epoch 10 --beta-dist`





