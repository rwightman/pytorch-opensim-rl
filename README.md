# PyTorch OpenSim Reinforcement Learning for CrowdAI Prosthetics Environment

This is my code for experimenting with the CrowdAI Prosthetics Challenge (https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge)

The reinforcement learning codebase is based upon Ilya Kostrikov's awesome work (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

As this is part of my learning process for continuous control with deep reinforcement learning, there are likely to be some issues.

Additions:
 * add support for the OpenSim Gym-like environments with Ilya's RL codebase
 * add custom 'MyProstheticsEnv' wrapper to allow easier experimentation with different observation projections, rewards, and other aspects
 * frame skipping support in custom env
 * beta distribution experiment for continuous control in the range [0, 1] (http://ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)
 * tweaks to logging/folders/checkpoints and model resume for easier experimentation and tracking of results
