from osim.env import ProstheticsEnv
import random
import numpy as np
import math
import os
import time
from collections import deque
import torch
import torch.nn.functional as F

PROJ_FULL = 0
PROJ_NORMAL = 1
PROJ_SIMPLE = 2


## Values in the observation vector
# y, vx, vy, ax, ay, rz, vrz, arz of pelvis (10 values)
# x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (12*6 values)
# rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 values)
# activation, fiber_len, fiber_vel for all muscles (3*18)
# x, y, vx, vy, ax, ay ofg center of mass (6)
# 8 + 9*6 + 8*3 + 3*18 + 6 = 146
def project_obs(state_desc, proj=PROJ_FULL, prosthetic=True):
    res = []

    if proj == PROJ_SIMPLE:
        pelvis = state_desc["body_pos"]["pelvis"][0:3]
        # pelvis_vel = state_desc["body_vel"]["pelvis"][0:3]
        # pelvis_acc = state_desc["body_acc"]["pelvis"][0:3]
        res += pelvis[1:2]  # + pelvis_vel[:] + pelvis_acc[:]
        for bp in ["talus_l", "pros_foot_r"]:
            bp_pos = state_desc["body_pos"][bp].copy()
            bp_pos[0] = bp_pos[0] - pelvis[0]
            bp_pos[2] = bp_pos[2] - pelvis[2]
            res += bp_pos
    else:
        pelvis = None
        for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            if prosthetic and body_part in ["toes_r", "talus_r"]:
                if proj == PROJ_FULL:
                    res += [0] * 12
                continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:3]
            cur += state_desc["body_vel"][body_part][0:3]
            cur += state_desc["body_acc"][body_part][0:3]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur.copy()
                res += pelvis[1:2] + pelvis[3:]
            else:
                cur_upd = cur.copy()
                cur_upd[:3] = [cur[i] - pelvis[i] for i in range(3)]
                cur_upd[9:10] = [cur[i] - pelvis[i] for i in range(9, 10)]
                res += cur_upd

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r', 'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                   'glut_max_l', 'glut_max_r', 'hamstrings_l', 'hamstrings_r', 'iliopsoas_l', 'iliopsoas_r',
                   'rect_fem_l', 'rect_fem_r', 'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    cm_vel = state_desc["misc"]["mass_center_vel"]
    cm_acc = state_desc["misc"]["mass_center_acc"]
    res = res + cm_pos + cm_vel + cm_acc

    return np.array(res)


def project_obs_mirror(state_desc, proj=PROJ_FULL, prosthetic=True):
    assert proj == PROJ_SIMPLE # FIXME only supporting simple projection right now

    res = []

    if proj == PROJ_SIMPLE:
        pelvis = state_desc["body_pos"]["pelvis"][0:3]
        # pelvis_vel = state_desc["body_vel"]["pelvis"][0:3]
        # pelvis_acc = state_desc["body_acc"]["pelvis"][0:3]
        res += pelvis[1:2]  # + pelvis_vel[:] + pelvis_acc[:]
        for bp in ["pros_foot_r", "talus_l"]:
            bp_pos = state_desc["body_pos"][bp].copy()
            bp_pos[0] = bp_pos[0] - pelvis[0]
            bp_pos[2] = bp_pos[2] - pelvis[2]
            print(bp_pos)
            res += bp_pos
    else:
        pelvis = None
        for body_part in ["pelvis", "head", "torso", "toes_r", "toes_l", "talus_r", "talus_l"]:
            if prosthetic and body_part in ["toes_r", "talus_r"]:
                # FIXME for reverse, would we use a mean? no symmetry here
                if proj == PROJ_FULL:
                    res += [0] * 12
                continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:3]
            cur += state_desc["body_vel"][body_part][0:3]
            cur += state_desc["body_acc"][body_part][0:3]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            # FIXME for mirror, more thought would be needed for the rotations of core vs leg parts
            if body_part == "pelvis":
                pelvis = cur.copy()
                res += pelvis[1:2] + pelvis[3:]
            else:
                cur_upd = cur.copy()
                cur_upd[:3] = [cur[i] - pelvis[i] for i in range(3)]
                cur_upd[9:10] = [cur[i] - pelvis[i] for i in range(9, 10)]
                res += cur_upd

    for joint in ["ankle_r", "ankle_l", "back", "hip_r", "hip_l", "knee_r", "knee_l"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in ['abd_r', 'abd_l', 'add_r', 'add_l', 'bifemsh_r', 'bifemsh_l', 'gastroc_l',
                   'glut_max_r', 'glut_max_l', 'hamstrings_r', 'hamstrings_l', 'iliopsoas_r', 'iliopsoas_l',
                   'rect_fem_r', 'rect_fem_l', 'soleus_l', 'tib_ant_l', 'vasti_r', 'vasti_l']:
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    cm_vel = state_desc["misc"]["mass_center_vel"]
    cm_acc = state_desc["misc"]["mass_center_acc"]
    res = res + cm_pos + cm_vel + cm_acc

    return np.array(res)


class ProsthetiscMirror:
    def __init__(self, device='cpu'):
        self.obs_mirror_idx = torch.tensor([
            0,  # pelvis h
            4, 5, 6,  # foot pos
            1, 2, 3,  # foot pos

            10, 11, 12,  # ankle r j  1
            7, 8, 9,  # ankle l j
            13, 14, 15,  # back  1
            25, 26, 27, 28, 29, 30, 31, 32, 33,  # hip r j 3
            16, 17, 18, 19, 20, 21, 22, 23, 24,  # hip l j
            37, 38, 39,  # knee r j  1
            34, 35, 36,  # knee l j

            43, 44, 45,  # abd_r m
            40, 41, 42,  # abd_l m
            49, 50, 51,  # add_r m
            46, 47, 48,  # add_l m
            55, 56, 57,  # bifmesh_r m
            52, 53, 54,  # bifmesh_l m
            58, 59, 60,  ##gastroc_l
            64, 65, 66,  # glut_max r m
            61, 62, 63,  # glut_max_l m
            70, 71, 72,  # hmstrings_r
            67, 68, 69,  # hamstrings_l
            76, 77, 78,  # iliopsoas_r
            73, 74, 75,  # iliopsoas_l
            82, 83, 84,  # rect_fem_r m
            79, 80, 81,  # rect_fem_l m
            85, 86, 87,  ## soleus_l m
            88, 89, 90,  ## tib_ant_l m
            94, 95, 96,  # vasti_r_m
            91, 92, 93,  # vasti_l m
            97, 98, 99,  # cm pos
            100, 101, 102,
            103, 104, 105,
            ]).to(device)

        self.action_idx = torch.tensor(
            [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]).to(device)

        self.action_mirror_idx = torch.tensor(
            [1, 0, 3, 2, 5, 4, 8, 7, 10, 9, 12, 11, 14, 13, 18, 17]).to(device)

    def mirror_obs(self, a):
        return torch.index_select(a, 1, self.obs_mirror_idx)

    def compare_actions(self, normal, mirror):
        a = torch.index_select(
            normal, 1,  self.action_idx)
        b = torch.index_select(
            mirror, 1, self.action_mirror_idx)
        mirror_loss = F.mse_loss(a, b)
        return mirror_loss

    def calc_loss(self, actor_critic, obs, recurrent_hidden_states, actions, masks):
        # Mirror loss
        actions_mirror = actor_critic.get_action(
            self.mirror_obs(obs),
            recurrent_hidden_states,
            masks)

        return self.compare_actions(actions, actions_mirror)


class MyProstheticsEnv(ProstheticsEnv):

    def __init__(self, visualize=False, integrator_accuracy=1e-4, difficulty=0, seed=0, frame_skip=0):
        self.project_mode = PROJ_SIMPLE
        super(MyProstheticsEnv, self).__init__(
            visualize=visualize,
            integrator_accuracy=integrator_accuracy,
            difficulty=difficulty,
            seed=seed)
        if difficulty == 0:
            self.time_limit = 600  # longer time limit to reduce likelihood of diving strategy
        self.spec.timestep_limit = self.time_limit
        np.random.seed(seed)
        self.frame_times = deque(maxlen=100)
        self.frame_count = 0
        self.frame_skip = frame_skip
        self.debug = False

        #self.muscle_names = [self.osim_model.muscleSet.get(i).getName() for i in range(self.osim_model.muscleSet.getSize())]
        #self.muscle_names = sorted(self.muscle_names)
        #print(self.muscle_names)

    def get_observation(self):
        state_desc = self.get_state_desc()
        return project_obs(state_desc, proj=self.project_mode, prosthetic=self.prosthetic)

    def get_observation_space_size(self):
        if self.prosthetic:
            if self.project_mode == PROJ_SIMPLE:
                return 106
            elif self.project_mode == PROJ_FULL:
                return 181
            else:
                return 157
        return 167

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.65

    def my_reward_round1(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        penalty = 0.
        penalty += (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][2]) ** 2
        penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
        if state_desc["body_pos"]["pelvis"][1] < 0.70:
            penalty += 10  # penalize falling more

        # Reward for not falling
        reward = 10.0

        return reward - penalty

    def my_reward_round2(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        penalty = 0

        # Small penalty for too much activation (cost of transport)
        penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001

        # Big penalty for not matching the vector on the X,Z projection.
        # No penalty for the vertical axis
        penalty += (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) ** 2
        if state_desc["body_pos"]["pelvis"][1] < 0.70:
            penalty += 10  # penalize falling more

        # Reward for not falling
        reward = 10.0

        return reward - penalty

    def reward_round1(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0)**2

    def reward_round2(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        penalty = 0

        # Small penalty for too much activation (cost of transport)
        penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001

        # Big penalty for not matching the vector on the X,Z projection.
        # No penalty for the vertical axis
        penalty += (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) ** 2

        # Reward for not falling
        reward = 10.0

        return reward - penalty

    def reward(self):
        if self.difficulty == 0:
            return self.reward_round1()
        return self.reward_round2()

    def my_reward(self):
        if self.difficulty == 0:
            return self.my_reward_round1()
        return self.my_reward_round2()

    def step(self, action, project=True):
        reward = 0.
        rewardb = 0.
        done = False

        if self.frame_skip:
            num_steps = self.frame_skip
        else:
            num_steps = 1

        for _ in range(num_steps):
            self.prev_state_desc = self.get_state_desc()

            start_time = time.perf_counter()
            self.osim_model.actuate(action)
            self.osim_model.integrate()
            step_time = time.perf_counter() - start_time

            # track some step stats across resets
            self.frame_times.append(step_time)
            self.frame_count += 1

            if self.debug and self.frame_count % 1000 == 0:
                frame_mean = np.mean(self.frame_times)
                frame_min = np.min(self.frame_times)
                frame_max = np.max(self.frame_times)
                print('Steps {}, duration mean, min, max: {:.3f}, {:.3f}, {:.3f}'.format(
                    self.frame_count, frame_mean, frame_min, frame_max))

            done = self.is_done() or self.osim_model.istep >= self.spec.timestep_limit
            if step_time > 15.:
                reward += -10
                done = True
            else:
                reward += self.my_reward()
            rewardb += self.reward()

            if done:
                break

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()

        return [obs, reward, done, {'rb': rewardb}]

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
