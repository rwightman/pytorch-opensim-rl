from osim.env import ProstheticsEnv
import random
import numpy as np
import math
import os


class MyProstheticsEnv(ProstheticsEnv):

    def __init__(self, visualize=False, integrator_accuracy=1e-4, difficulty=0, seed=0):
        super(MyProstheticsEnv, self).__init__(
            visualize=visualize,
            integrator_accuracy=integrator_accuracy,
            difficulty=difficulty,
            seed=seed)
        if difficulty == 0:
            self.time_limit = 600  # longer time limit to reduce likelihood of diving strategy
        np.random.seed(seed)

    ## Values in the observation vector
    # y, vx, vy, ax, ay, rz, vrz, arz of pelvis (10 values)
    # x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (12*6 values)
    # rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 values)
    # activation, fiber_len, fiber_vel for all muscles (3*18)
    # x, y, vx, vy, ax, ay ofg center of mass (6)
    # 8 + 9*6 + 8*3 + 3*18 + 6 = 146
    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            if self.prosthetic and body_part in ["toes_r", "talus_r"]:
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
                pelvis = cur
                blorb = cur[1:2] + cur[3:]
                res += blorb
            else:
                cur_upd = cur.copy()
                cur_upd[:3] = [cur[i] - pelvis[i] for i in range(3)]
                cur_upd[9:10] = [cur[i] - pelvis[i] for i in range(9, 10)]
                res += cur_upd

        for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
        cm_vel = state_desc["misc"]["mass_center_vel"]
        cm_acc = state_desc["misc"]["mass_center_acc"]
        #if self.difficulty == 0:
        #    cm_pos = cm_pos[0:2]
        #    cm_vel = cm_vel[0:2]
        #    cm_acc = cm_acc[0:2]
        res = res + cm_pos + cm_vel + cm_acc

        return np.array(res)

    def get_observation_space_size(self):
        if self.prosthetic == True:
            return 181
        return 167

    def my_reward_round1(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        penalty = 0.
        penalty += (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][2]) ** 2
        penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001

        return 10. - penalty

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
        self.prev_state_desc = self.get_state_desc()
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()

        done = self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit)

        return [obs, self.my_reward(), done, {'or': self.reward()}]

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)