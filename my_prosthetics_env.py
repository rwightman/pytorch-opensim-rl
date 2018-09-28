from osim.env import ProstheticsEnv
import random
import numpy as np
import math
import os


class MyProstheticsEnv(ProstheticsEnv):

    def __init__(self, visualize = True, integrator_accuracy = 5e-5, difficulty=0, seed=0):
        super(MyProstheticsEnv, self).__init__(
            visualize=visualize, integrator_accuracy=integrator_accuracy, difficulty=difficulty, seed=seed)
        np.random.seed(seed)

    ## Values in the observation vector
    # y, vx, vy, ax, ay, rz, vrz, arz of pelvis (8 values)
    # x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (9*6 values)
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
                res += [0] * 9
                continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                cur_upd = cur
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6, 7)]
                res += cur

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
        if self.difficulty == 0:
            cm_pos = cm_pos[0:2]
            cm_vel = cm_vel[0:2]
            cm_acc = cm_acc[0:2]
        res = res + cm_pos + cm_vel + cm_acc

        return res

    def get_observation_space_size(self):
        if self.prosthetic == True:
            return 158
        return 167

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
