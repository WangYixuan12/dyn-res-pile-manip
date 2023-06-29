import sys
sys.path.append('.')

from typing import List
import cv2
import gym
import numpy as np
import torch
from utils import load_yaml, fps, fps_rad, recenter, depth2fgpcd, pcd2pix, fps_np, downsample_pcd
import time
import pyflex
import os
import pybullet as p
import pybullet_data
from bs4 import BeautifulSoup
from transformations import rotation_matrix, quaternion_from_matrix, quaternion_matrix
from planners import PlannerGD
from env.flex_rewards import config_reward, config_reward_ptcl
import matplotlib.pyplot as plt
import math

from scipy.special import softmax
from model.res_regressor import MPCResRgrNoPool

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A,B,C,D):
    res = ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    return res

def proc_obs(obs, config, res=64):
    global_scale = config['dataset']['global_scale']
    assert type(obs) == np.ndarray
    assert obs.shape[-1] == 5
    assert obs[..., :3].max() <= 255.0
    assert obs[..., :3].min() >= 0.0
    assert obs[..., :3].max() >= 1.0
    assert obs[..., -1].max() >= 0.7 * global_scale
    assert obs[..., -1].max() <= 0.8 * global_scale

    obs[..., :3] = obs[..., :3][..., ::-1] / 255.
    obs[..., -1] = obs[..., -1] / (global_scale)
    obs = np.concatenate([obs[..., :3], obs[..., -1:]], axis=-1)
    obs[obs[..., -1] > 0.599/0.8, :3] = 1.0
    obs = cv2.resize(obs, (res, res), interpolation=cv2.INTER_AREA)
    
    # binary
    obs = (obs[..., -1] <= 0.599/0.8).astype(np.float32)
    obs = obs[..., None]

    assert type(obs) == np.ndarray
    assert obs.shape == (res, res, 1) or obs.shape == (res, res, 4)
    return obs

# assumption:
#  - all robots are in pybullet data folder
#  - only load one robot
class FlexRobotHelper:
    def __init__(self):
        self.transform_bullet_to_flex = np.array([
            [1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        self.robotId = None

    def loadURDF(self, fileName, basePosition, baseOrientation, useFixedBase = True, globalScaling = 1.0):
        if self.robotId is None:
            self.robotId = p.loadURDF(fileName, basePosition, baseOrientation, useFixedBase = useFixedBase, globalScaling = globalScaling)
        p.resetBasePositionAndOrientation(self.robotId, basePosition, baseOrientation)
        root_path = pybullet_data.getDataPath()
        robot_path = os.path.join(root_path, fileName)
        robot_path_par = os.path.abspath(os.path.join(robot_path, os.pardir))
        with open(robot_path, 'r') as f:
            robot = f.read()
        robot_data = BeautifulSoup(robot, 'xml')
        links = robot_data.find_all('link')
        self.num_meshes = 0
        self.has_mesh = np.ones(len(links), dtype=bool)
        for i in range(len(links)):
            link = links[i]
            if link.find_all('geometry'):
                mesh_name = link.find_all('geometry')[0].find_all('mesh')[0].get('filename')
                if mesh_name[-4:] == '.STL':
                    mesh_name = mesh_name[:-4] + '.obj'
                pyflex.add_mesh(os.path.join(robot_path_par, mesh_name), globalScaling, 0, np.ones(3))
                self.num_meshes += 1
            else:
                self.has_mesh[i] = False
        
        self.num_link = len(links)
        self.state_pre = None

        return self.robotId

    def resetJointState(self, i, pose):
        p.resetJointState(self.robotId, i, pose)
        return self.getRobotShapeStates()
    
    def getRobotShapeStates(self):
        # convert pybullet link state to pyflex link state
        state_cur = []
        base_com_pos, base_com_orn = p.getBasePositionAndOrientation(self.robotId)
        di = p.getDynamicsInfo(self.robotId, -1)
        local_inertial_pos, local_inertial_orn = di[3], di[4]
        pos_inv, orn_inv = p.invertTransform(local_inertial_pos, local_inertial_orn)
        pos, orn = p.multiplyTransforms(base_com_pos, base_com_orn, pos_inv, orn_inv)
        state_cur.append(list(pos) + [1] + list(orn))

        for l in range(self.num_link-1):
            ls = p.getLinkState(self.robotId, l)
            pos = ls[4]
            orn = ls[5]
            state_cur.append(list(pos) + [1] + list(orn))
        
        state_cur = np.array(state_cur)
        shape_states = np.zeros((self.num_meshes, 14))
        if self.state_pre is None:
            self.state_pre = state_cur.copy()

        mesh_idx = 0
        for i in range(self.num_link):
            if self.has_mesh[i]:
                shape_states[mesh_idx, 0:3] = np.matmul(
                    self.transform_bullet_to_flex, state_cur[i, :4])[:3]
                shape_states[mesh_idx, 3:6] = np.matmul(
                    self.transform_bullet_to_flex, self.state_pre[i, :4])[:3]
                shape_states[mesh_idx, 6:10] = quaternion_from_matrix(
                    np.matmul(self.transform_bullet_to_flex,
                            quaternion_matrix(state_cur[i, 4:])))
                shape_states[mesh_idx, 10:14] = quaternion_from_matrix(
                    np.matmul(self.transform_bullet_to_flex,
                            quaternion_matrix(self.state_pre[i, 4:])))
                mesh_idx += 1
        
        self.state_pre = state_cur
        return shape_states

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

class FlexEnv(gym.Env):
    def __init__(self, config=None) -> None:
        super().__init__()
        
        self.is_real = False

        # set up pybullet
        # physicsClient = p.connect(p.GUI)
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")

        # set up pyflex
        self.screenWidth = 720
        self.screenHeight = 720
        self.wkspc_w = config['dataset']['wkspc_w']
        self.headless = config['dataset']['headless']
        self.obj = config['dataset']['obj']
        self.global_scale = config['dataset']['global_scale']
        self.cont_motion = config['dataset']['cont_motion']
        self.init_pos = config['dataset']['init_pos']
        self.robot_type = config['dataset']['robot_type']
        self.img_channel = 1
        self.config = config

        pyflex.set_screenWidth(self.screenWidth)
        pyflex.set_screenHeight(self.screenHeight)
        pyflex.set_light_dir(np.array([0.1, 2.0, 0.1]))
        pyflex.set_light_fov(70.)
        pyflex.init(config['dataset']['headless'])

        # set up camera
        cam_idx = config['dataset']['cam_idx']
        rad = np.deg2rad(cam_idx * 20.)
        cam_dis = 0.0 * self.global_scale / 8.0
        # cam_dis = 6.0 * self.global_scale / 8.0
        cam_height = 6.0 * self.global_scale / 8.0
        # cam_height = 4.0 * self.global_scale / 8.0
        self.camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
        self.camAngle = np.array([rad, -np.deg2rad(90.), 0.])
        # self.camAngle = np.array([rad, -np.deg2rad(25.), 0.])

        # define robot information
        self.flex_robot_helper = FlexRobotHelper()
        if self.robot_type == 'franka':
            self.end_idx = 11
            self.num_dofs = 9
            self.left_finger_joint = 9
            self.right_finger_joint = 10
        elif self.robot_type == 'kinova':
            self.end_idx = 7
            self.num_dofs = 7

        # define action space
        self.act_dim = 4

    def robot_to_shape_states(self, robot_states):
        return np.concatenate([self.wall_shape_states, robot_states], axis=0)

    def reset_panda(self, jointPositions = np.zeros(9).tolist()):
        index = 0
        for j in range(self.num_joints):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)

            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                # p.setJointMotorControl2(bodyIndex=self.robotId,
                #                         jointIndex=j,
                #                         controlMode=p.POSITION_CONTROL,
                #                         targetPosition=jointPositions[index],
                #                         targetVelocity=0)
                if self.robot_type == 'franka':
                    if index < self.num_dofs - 2:
                        pyflex.resetJointState(self.flex_robot_helper, j, jointPositions[index])
                    else:
                        pyflex.resetJointState(self.flex_robot_helper, j, 0.)
                elif self.robot_type == 'kinova':
                    pyflex.resetJointState(self.flex_robot_helper, j, jointPositions[index])
                else:
                    raise NotImplementedError
                index=index+1
            # p.stepSimulation()
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
        # p.resetJointState(self.robotId, self.right_finger_joint, 0.0)
        # p.resetJointState(self.robotId, self.left_finger_joint, 0.0)

    def step(self, action, video_recorder=None, add_cam_idx=None):
        # Inputs:
        #   - action: a 4d numpy array with following fields
        #     - action[0-1] - starting position
        #     - action[2-3] - ending position
        # Outputs:
        #  - obs: an RGBD image
        if self.act_dim == 4:
            if self.robot_type == 'franka':
                h = self.global_scale / 8.0
            elif self.robot_type == 'kinova':
                h = 0.11 * self.global_scale
            else:
                raise NotImplementedError
            s_2d = np.concatenate([action[:2], [h]])
            e_2d = np.concatenate([action[2:], [h]])
        elif self.act_dim == 6:
            s_2d = action[:3]
            e_2d = action[3:]
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi/2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1]/(s_2d - e_2d)[0])
        if self.robot_type == 'franka':
            orn = np.array([0.0, np.pi, pusher_angle+np.pi/2])
        elif self.robot_type == 'kinova':
            orn = np.array([0.0, np.pi, pusher_angle])
        # halfEdge = np.array([0.05, 1.0, 0.4])
        # quat = quatFromAxisAngle(
        #     axis=np.array([0., 1., 0.]),
        #     angle=-pusher_angle)

        if self.cont_motion:
            # curr_pt = p.getJointState(self.robotId, self.end_idx)[0]
            if self.last_ee is None:
                self.reset_panda(self.rest_joints)
                self.last_ee = s_2d + np.array([0., 0., self.global_scale / 6.0])
            way_pts = [self.last_ee, s_2d + np.array([0., 0., self.global_scale / 6.0]), s_2d, e_2d, e_2d + np.array([0., 0., self.global_scale / 6.0]), e_2d + np.array([-self.global_scale/3.0-e_2d[0], 0., self.global_scale / 6.0])]
        else:
            way_pts = [s_2d + np.array([0., 0., self.global_scale / 24.0]), s_2d, e_2d, e_2d + np.array([0., 0., self.global_scale / 24.0])]
            self.reset_panda(self.rest_joints)
        speed = 1.0/50.
        for i_p in range(len(way_pts)-1):
            s = way_pts[i_p]
            e = way_pts[i_p+1]
            steps = int(np.linalg.norm(e-s)/speed) + 1
            for i in range(steps):
                end_effector_pos = s + (e - s) * i / steps
                end_effector_orn = p.getQuaternionFromEuler(orn)
                jointPoses = p.calculateInverseKinematics(self.robotId,
                                                        self.end_idx,
                                                        end_effector_pos,
                                                        end_effector_orn,
                                                        self.joints_lower.tolist(),
                                                        self.joints_upper.tolist(),
                                                        (self.joints_upper - self.joints_lower).tolist(),
                                                        self.rest_joints)
                self.reset_panda(jointPoses)
                # obs = pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
                # pyflex.render(render_depth=True)
                if video_recorder:
                    obs = self.render(add_cam_idx=add_cam_idx)
                    if not isinstance(obs, list):
                        video_recorder[0].write(obs[..., :3][..., ::-1].astype(np.uint8))
                    else:
                        for i, obs_i in enumerate(obs):
                            video_recorder[i].write(obs_i[..., :3][..., ::-1].astype(np.uint8))
                pyflex.step()
                if math.isnan(self.get_positions().reshape(-1, 4)[:, 0].max()):
                    print('simulator exploded when action is ', action)
                    return None
            self.last_ee = end_effector_pos.copy()
        if not self.cont_motion:
            self.reset_panda()
        for i in range(200):
            if video_recorder:
                obs = self.render(add_cam_idx=add_cam_idx)
                if not isinstance(obs, list):
                    video_recorder[0].write(obs[..., :3][..., ::-1].astype(np.uint8))
                else:
                    for i, obs_i in enumerate(obs):
                        video_recorder[i].write(obs_i[..., :3][..., ::-1].astype(np.uint8))
            pyflex.step()
        obs = self.render(add_cam_idx=add_cam_idx)

        return obs
    
    def clip_action(self, action):
        # cont_points = [np.array([-self.cont_half_w, self.cont_half_w]),
        #                np.array([-self.cont_half_w, -self.cont_half_w]),
        #                np.array([self.cont_half_w, -self.cont_half_w]),
        #                np.array([self.cont_half_w, self.cont_half_w])]
        cont_points = [np.array([self.cont_centers[1][0], -self.cont_centers[1][2]+self.cont_half_w]),
                       np.array([self.cont_centers[1][0], -self.cont_centers[1][2]-self.cont_half_w]),
                       np.array([self.cont_centers[0][0], -self.cont_centers[0][2]-self.cont_half_w]),
                       np.array([self.cont_centers[0][0], -self.cont_centers[0][2]+self.cont_half_w])]
        shift_arr = np.array([0, 0])
        if self.act_dim == 4:
            s_2d = action[:2]
            e_2d = action[2:]
            h = self.global_scale / 8.0
        elif self.act_dim == 6:
            s_2d = action[:3]
            e_2d = action[3:]
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi/2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1]/(s_2d - e_2d)[0])
        speed = 1.0/50.
        steps = int(np.linalg.norm(e_2d-s_2d)/speed) + 1
        for i in range(steps):
            if self.act_dim == 4:
                end_effector_pos = np.concatenate([s_2d + (e_2d - s_2d) * i / steps, [h]])
            elif self.act_dim == 6:
                end_effector_pos = s_2d + (e_2d - s_2d) * i / steps
            for j in range(3):
                pusher_w = 0.05 * self.global_scale
                left_p = end_effector_pos[:2]+np.array([pusher_w*np.cos(pusher_angle-np.pi/2), pusher_w*np.sin(pusher_angle-np.pi/2)])
                right_p = end_effector_pos[:2]+np.array([-pusher_w*np.cos(pusher_angle-np.pi/2), -pusher_w*np.sin(pusher_angle-np.pi/2)])
                if intersect(left_p, right_p, cont_points[j], cont_points[j+1]) or intersect(left_p, right_p, cont_points[j]-shift_arr, cont_points[j+1]-shift_arr):
                    if i <= 15:
                        return None
                    else:
                        action[self.act_dim//2:] = s_2d + (e_2d - s_2d) * (i-15) / steps
                        return action
        return action

    def sample_action_obj_biased(self, n):
        particles = self.get_positions()
        num_particles = particles.shape[0] // 4
        particles = particles.reshape(num_particles, 4)
        rand_idx = np.random.choice(num_particles, n, replace=False)
        rand_particles = particles[rand_idx]
        start_center = np.zeros((n, 2))
        start_center[:, 0] = rand_particles[:, 0]
        start_center[:, 1] = -rand_particles[:, 2]
        sigma = 0.5 * self.global_scale / 12.0
        start_center += np.random.normal(0, sigma, size=start_center.shape)
        actions = np.zeros((n, self.act_dim))
        actions[:, :2] = np.clip(start_center, -self.wkspc_w, self.wkspc_w)
        actions[:, 2:4] = np.random.uniform(-self.wkspc_w, self.wkspc_w, n)
        return actions

    def sample_action(self, n):
        # sample one action within feasible space and with corresponding convex region label
        action = -self.wkspc_w + 2 * self.wkspc_w * np.random.rand(n, 1, 4)
        reg_label = np.zeros(n)
        return action, reg_label
    
    def sample_particle_center(self, n):
        self.cvx_region = np.zeros((1,4)) # every row: left, right, bottom, top
        self.cvx_region[0,0] = -self.wkspc_w
        self.cvx_region[0,1] = self.wkspc_w
        self.cvx_region[0,2] = -self.wkspc_w
        self.cvx_region[0,3] = self.wkspc_w
        centers = -self.wkspc_w + 2 * self.wkspc_w * np.random.rand(n, 2)
        reg_label = np.zeros(n)
        return centers

    def reset(self):
        if self.obj == 'coffee':
            scale = 0.2 * self.global_scale / 8.0
            x = -0.9 * self.global_scale / 8.0
            y = 0.5
            z = -0.9 * self.global_scale / 8.0
            staticFriction = 0.0
            dynamicFriction = 1.0
            draw_skin = 1.0
            num_coffee = 1000 # [200, 1000]
            self.scene_params = np.array([
                scale, x, y, z, staticFriction, dynamicFriction, draw_skin, num_coffee])
            pyflex.set_scene(20, self.scene_params, 0)
        elif self.obj == 'ball':
            scale = 0.7
            x = -scale/2
            y = 0.0
            z = -scale/2
            self.init_x = x + scale/2
            self.init_y = y + scale/2
            self.init_z = z + scale/2
            staticFriction = 1.0
            dynamicFriction = 0.7
            particle_radius = self.config['dataset']['particle_r']
            self.scene_params = np.array([
                scale, x, y, z, staticFriction, dynamicFriction, particle_radius])
            pyflex.set_scene(18, self.scene_params, 0)
        elif self.obj == 'capsule':
            scale = 0.2 * self.global_scale / 8.0
            x = -1. * self.global_scale / 8.0
            y = 0.5
            z = -1. * self.global_scale / 8.0
            staticFriction = 0.0
            dynamicFriction = 0.5
            draw_skin = 1.0
            num_capsule = 200 # [200, 1000]
            slices = 10
            segments = 20
            self.scene_params = np.array([
                scale, x, y, z, staticFriction, dynamicFriction, draw_skin, num_capsule, slices, segments])
            pyflex.set_scene(21, self.scene_params, 0)
        elif self.obj == 'carrots':
            staticFriction = 1.0
            dynamicFriction = 0.9
            draw_skin = 1.0
            min_dist = 10.0
            max_dist = 20.0
            self.cvx_region = np.zeros((1,4)) # every row: left, right, bottom, top
            self.cvx_region[0,0] = -self.wkspc_w
            self.cvx_region[0,1] = self.wkspc_w
            self.cvx_region[0,2] = -self.wkspc_w
            self.cvx_region[0,3] = self.wkspc_w
            if self.init_pos == 'spread':
                max_scale = 0.1 * self.global_scale / 8.0
                min_scale = 0.1 * self.global_scale / 8.0
                x = -1.5 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.5 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2 + 1
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2 + 1
                num_carrots = (num_x * num_z - 1) * 3
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'wkspc_spread':
                max_scale = 0.2 * self.global_scale / 8.0
                min_scale = 0.2 * self.global_scale / 8.0
                x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2
                num_carrots = num_x * num_z - 1
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'wkspc_spread_double':
                max_scale = 0.2 * self.global_scale / 8.0
                min_scale = 0.2 * self.global_scale / 8.0
                x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2
                num_carrots = 2 * (num_x * num_z - 1)
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'wkspc_spread_triple':
                max_scale = 0.2 * self.global_scale / 8.0
                min_scale = 0.2 * self.global_scale / 8.0
                x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2
                num_carrots = 3 * (num_x * num_z - 1)
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'wkspc_spread_4':
                max_scale = 0.2 * self.global_scale / 8.0
                min_scale = 0.2 * self.global_scale / 8.0
                x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2
                num_carrots = 4 * (num_x * num_z - 1)
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'extra_large_wkspc_spread':
                max_scale = 0.3 * self.global_scale / 8.0
                min_scale = 0.3 * self.global_scale / 8.0
                x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale) * 2
                num_carrots = 2 * (num_x * num_z - 1)
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'extra_small_wkspc_spread':
                max_scale = 0.09 * self.global_scale / 8.0
                min_scale = 0.09 * self.global_scale / 8.0
                x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2
                num_carrots = 4 * (num_x * num_z - 1)
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'extra_small_half_spread':
                max_scale = 0.09 * self.global_scale / 8.0
                min_scale = 0.09 * self.global_scale / 8.0
                x = -0.9 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -0.9 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2
                num_carrots = 4 * (num_x * num_z - 1)
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'rand_blob':
                rand_scale = np.random.uniform(0.07, 0.12) * self.global_scale / 8.0
                max_scale = rand_scale
                min_scale = rand_scale
                blob_r = np.random.uniform(0.3, 0.5)
                x = -blob_r * self.global_scale / 8.0
                y = 0.5
                z = -blob_r * self.global_scale / 8.0
                inter_space = max_scale
                num_x = int(abs(x) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z) / max_scale) * 2
                x_off = self.global_scale * np.random.uniform(-1./12., 1./8.)
                z_off = self.global_scale * np.random.uniform(-1./12., 1./8.)
                x += x_off
                z += z_off
                print('rand_scale: ', rand_scale)
                print('blob_r: ', blob_r)
                print('x_off: ', x_off)
                print('z_off: ', z_off)
                num_carrots = (num_x * num_z - 1) * 3
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'rand_spread':
                rand_scale = np.random.uniform(0.09, 0.12) * self.global_scale / 8.0
                max_scale = rand_scale
                min_scale = rand_scale
                blob_r = np.random.uniform(0.7, 1.0)
                x = - blob_r * self.global_scale / 8.0
                y = 0.5
                z = - blob_r * self.global_scale / 8.0
                inter_space = 1.5 * max_scale
                num_x = int(abs(x/1.5) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/1.5) / max_scale + 1) * 2
                x_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
                z_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
                x += x_off
                z += z_off
                num_carrots = (num_x * num_z - 1) * 3
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'rand_sparse_spread':
                rand_scale = 0.12 * self.global_scale / 8.0
                max_scale = rand_scale
                min_scale = rand_scale
                blob_r = np.random.uniform(1.0, 1.5)
                x = - blob_r * self.global_scale / 8.0
                y = 0.5
                z = - blob_r * self.global_scale / 8.0
                inter_space = max_scale * 2
                num_x = int(abs(x/2.) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale) * 2
                # x_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
                # z_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
                # x += x_off
                # z += z_off
                num_carrots = (num_x * num_z - 1) * 1
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'rb_corner':
                max_scale = 0.12 * self.global_scale / 8.0
                min_scale = 0.12 * self.global_scale / 8.0
                x = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = max_scale
                num_x = int(abs(x) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z) / max_scale) * 2
                x += self.global_scale / 8.
                z += self.global_scale / 8.
                num_carrots = (num_x * num_z - 1) * 3
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'center':
                max_scale = 0.12 * self.global_scale / 8.0
                min_scale = 0.12 * self.global_scale / 8.0
                x = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = max_scale
                num_x = int(abs(x) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z) / max_scale) * 2
                num_carrots = (num_x * num_z - 1) * 3
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'center_init_2':
                max_scale = 0.12 * self.global_scale / 8.0
                min_scale = 0.12 * self.global_scale / 8.0
                x = -1.0 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.0 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = max_scale * 2
                num_x = int(abs(x/2.) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale) * 2
                num_carrots = (num_x * num_z - 1) * 1
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 1.0
            elif self.init_pos == 'rt_corner':
                max_scale = 0.15 * self.global_scale / 8.0
                min_scale = 0.15 * self.global_scale / 8.0
                x = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = max_scale
                num_x = int(abs(x) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z) / max_scale) * 2
                x += self.global_scale / 8.
                z -= self.global_scale / 8.
                num_carrots = int(0.25 * self.global_scale / (max_scale ** 2))
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'wkspc_spread_multi_granularity':
                max_scale = 0.2 * self.global_scale / 8.0
                min_scale = 0.05 * self.global_scale / 8.0
                x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = 2 * max_scale
                num_x = int(abs(x/2.) / max_scale + 1) * 2
                num_y = 10
                num_z = int(abs(z/2.) / max_scale + 1) * 2
                num_carrots = (num_x * num_z - 1) * 2
                add_singular = 0.0
                add_sing_x = -1
                add_sing_y = -1
                add_sing_z = -1
                add_noise = 0.0
            elif self.init_pos == 'singular':
                max_scale = 0.15 * self.global_scale / 8.0
                min_scale = 0.15 * self.global_scale / 8.0
                x = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = max_scale
                num_x = int(abs(x) / max_scale) * 2
                num_y = 10
                num_z = int(abs(z) / max_scale) * 2
                x -= self.global_scale / 8.
                num_carrots = int(0.25 * self.global_scale / (max_scale ** 2))
                add_singular = 1.0
                add_sing_x = 3.0 * self.global_scale / 24.0
                add_sing_y = 0.5
                add_sing_z = 0.0
                add_noise = 0.0
            elif self.init_pos == 'blank':
                max_scale = 0.15 * self.global_scale / 8.0
                min_scale = 0.15 * self.global_scale / 8.0
                x = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
                y = 0.5
                z = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
                inter_space = max_scale
                num_x = 1
                num_y = 10
                num_z = 1
                x -= self.global_scale
                num_carrots = 1
                add_singular = 0.0
                add_sing_x = 3.0 * self.global_scale / 24.0
                add_sing_y = 0.5
                add_sing_z = 0.0
                add_noise = 0.0
            else:
                raise NotImplementedError
            self.scene_params = np.array([max_scale,
                                          min_scale,
                                          x,
                                          y,
                                          z,
                                          staticFriction,
                                          dynamicFriction,
                                          draw_skin,
                                          num_carrots,
                                          min_dist,
                                          max_dist,
                                          num_x,
                                          num_y,
                                          num_z,
                                          inter_space,
                                          add_singular,
                                          add_sing_x,
                                          add_sing_y,
                                          add_sing_z,
                                          add_noise,])
            pyflex.set_scene(22, self.scene_params, 0)
        elif self.obj == 'coffee_capsule':
            cof_scale = 0.2 * self.global_scale / 8.0
            cof_x = -1.5 * self.global_scale / 8.0
            cof_y = 0.5
            cof_z = -1.2 * self.global_scale / 8.0
            staticFriction = 0.0
            dynamicFriction = 1.0
            draw_skin = 1.0
            num_coffee = 100 # [100, 300]
            num_capsule = 200 # [200, 500]
            cap_scale = 0.2 * self.global_scale / 8.0
            cap_x = 0. * self.global_scale / 8.0
            cap_y = 0.5
            cap_z = -1.2 * self.global_scale / 8.0
            cap_slices = 10
            cap_segments = 20
            self.scene_params = np.array([
                cof_scale, cof_x, cof_y, cof_z, staticFriction, dynamicFriction, draw_skin, num_coffee,
                cap_scale, cap_x, cap_y, cap_z, num_capsule, cap_slices, cap_segments])
            pyflex.set_scene(23, self.scene_params, 0)
        else:
            raise ValueError('obj not defined')

        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)

        for i in range(500):
            pyflex.step()

        # add wall
        halfEdge = np.array([0.05, 1.0, self.global_scale/2.0])
        centers = [np.array([self.global_scale/2.0, 1.0, 0.0]),
                   np.array([0.0, 1.0, -self.global_scale/2.0]),
                   np.array([-self.global_scale/2.0, 1.0, 0.0]),
                   np.array([0.0, 1.0, self.global_scale/2.0])]
        quats = [quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=0.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=np.pi/2.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=0.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]),
                                 angle=np.pi/2.)]
        hideShape = 0
        color = np.ones(3) * 0.9
        self.wall_shape_states = np.zeros((4, 14))
        for i, center in enumerate(centers):
            pyflex.add_box(halfEdge, center, quats[i], hideShape, color)
            self.wall_shape_states[i] = np.concatenate([center, center, quats[i], quats[i]])

        # add robot
        if self.robot_type == 'franka':
            self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'franka_panda/panda.urdf', [-4.5 * self.global_scale / 8.0, 0, 0], [0, 0, 0, 1], globalScaling=self.global_scale)
            self.rest_joints = [np.pi*5/8, -np.pi/2, -np.pi/2, -np.pi*5/8, -np.pi/4, np.pi/2, np.pi/4, 0., 0.]
        elif self.robot_type == 'kinova':
            self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'kinova/urdf/GEN3_URDF_V12.urdf', [-0.5 * self.global_scale, 0, 0], [0, 0, 0, 1], globalScaling=self.global_scale)
            self.rest_joints = [0., np.pi/6., np.pi, -np.pi/2., 0., -np.pi/3., -np.pi/4]
        else:
            raise NotImplementedError
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(self.robot_to_shape_states(pyflex.resetJointState(self.flex_robot_helper, idx, joint)))
        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.last_ee = None
        self.reset_panda()

    def render(self, no_return=False, add_cam_idx=None):
        # RGB scale: 0-255
        # Depth scale: float in meters
        pyflex.step()
        # color, depth = pyflex.render(render_depth=True)
        # img = np.concatenate([color.reshape(self.screenHeight, self.screenWidth, 4), depth.reshape(self.screenHeight, self.screenWidth, 1)], axis=2)
        # return img.reshape(self.screenHeight, self.screenWidth, 5)
        if no_return:
            return
        else:
            if add_cam_idx is None:
                return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
            else:
                imgs = []
                imgs.append(pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5))
                for cam_idx in add_cam_idx:
                    rad = np.deg2rad(cam_idx * 45.)
                    cam_dis = 7.0 * self.global_scale / 8.0
                    cam_height = 4.0 * self.global_scale / 8.0
                    camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
                    camAngle = np.array([rad, -np.deg2rad(25.), 0.])
                    pyflex.set_camPos(camPos)
                    pyflex.set_camAngle(camAngle)
                    imgs.append(pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5))
                pyflex.set_camPos(self.camPos)
                pyflex.set_camAngle(self.camAngle)
                return imgs
    
    def obs2ptcl(self, obs, particle_r):
        assert type(obs) == np.ndarray
        assert obs.shape[-1] == 5
        assert obs[..., :3].max() <= 255.0
        assert obs[..., :3].min() >= 0.0
        assert obs[..., :3].max() >= 1.0
        assert obs[..., -1].max() >= 0.7 * self.global_scale
        assert obs[..., -1].max() <= 0.8 * self.global_scale
        depth = obs[..., -1] / self.global_scale

        fgpcd = depth2fgpcd(depth, depth<0.599/0.8, self.get_cam_params())
        sampled_ptcl = fps_rad(fgpcd, particle_r)
        sampled_ptcl = recenter(fgpcd, sampled_ptcl, r = min(0.02, 0.5 * particle_r))
        return sampled_ptcl

    def obs2ptcl_fixed_num(self, obs, particle_num):
        assert type(obs) == np.ndarray
        assert obs.shape[-1] == 5
        assert obs[..., :3].max() <= 255.0
        assert obs[..., :3].min() >= 0.0
        assert obs[..., :3].max() >= 1.0
        assert obs[..., -1].max() >= 0.7 * self.global_scale
        assert obs[..., -1].max() <= 0.8 * self.global_scale
        depth = obs[..., -1] / self.global_scale

        fgpcd = depth2fgpcd(depth, depth<0.599/0.8, self.get_cam_params())
        fgpcd = downsample_pcd(fgpcd, 0.01)
        sampled_ptcl, particle_r = fps(fgpcd, particle_num)
        sampled_ptcl = recenter(fgpcd, sampled_ptcl, r = min(0.02, 0.5 * particle_r))
        return sampled_ptcl, particle_r

    def obs2ptcl_fixed_num_batch(self, obs, particle_num, batch_size):
        assert type(obs) == np.ndarray
        assert obs.shape[-1] == 5
        assert obs[..., :3].max() <= 255.0
        assert obs[..., :3].min() >= 0.0
        assert obs[..., :3].max() >= 1.0
        assert obs[..., -1].max() >= 0.7 * self.global_scale
        assert obs[..., -1].max() <= 0.8 * self.global_scale
        depth = obs[..., -1] / self.global_scale
        
        batch_sampled_ptcl = np.zeros((batch_size, particle_num, 3))
        batch_particle_r = np.zeros((batch_size, ))
        for i in range(batch_size):
            fgpcd = depth2fgpcd(depth, depth<0.599/0.8, self.get_cam_params())
            fgpcd = downsample_pcd(fgpcd, 0.01)
            sampled_ptcl, particle_r = fps(fgpcd, particle_num)
            batch_sampled_ptcl[i] = recenter(fgpcd, sampled_ptcl, r = min(0.02, 0.5 * particle_r))
            batch_particle_r[i] = particle_r
        return batch_sampled_ptcl, batch_particle_r

    def step_subgoal_ptcl(self,
                          subgoal,
                          model_dy,
                          init_pos = None,
                          n_mpc=30,
                          n_look_ahead=1,
                          n_sample=100,
                          n_update_iter=100,
                          gd_loop=1,
                        #   particle_r=0.06,
                          particle_num=50,
                          mpc_type='GD',
                          funnel_dist=None,
                          action_seq_mpc_init=None,
                          action_label_seq_mpc_init=None,
                          time_lim=float('inf'),
                          auto_particle_r=False,):
        assert type(subgoal) == np.ndarray
        assert subgoal.shape == (self.screenHeight, self.screenWidth)
        # planner
        if mpc_type == 'GD':
            self.planner = PlannerGD(self.config, self)
        else:
            raise NotImplementedError
        action_lower_lim = action_upper_lim = np.zeros(4) # DEPRECATED, should be of no use
        reward_params = (self.get_cam_extrinsics(), self.get_cam_params(), self.global_scale)

        particle_den_seq = []
        if auto_particle_r:
            res_rgr_folder = self.config['mpc']['res_sel']['model_folder']
            res_rgr_folder = os.path.join('data/res_regressor', res_rgr_folder)
            res_rgr = MPCResRgrNoPool(self.config)
            # res_rgr = MPCResCls(self.config)
            if self.config['mpc']['res_sel']['iter_num'] == -1:
                res_rgr.load_state_dict(torch.load(os.path.join(res_rgr_folder, 'net_best_dy_state_dict.pth')))
            else:
                res_rgr.load_state_dict(torch.load(os.path.join(res_rgr_folder, 'net_dy_iter_%d_state_dict.pth' % self.config['mpc']['res_sel']['iter_num'])))
            res_rgr = res_rgr.cuda()
            
            # construct res_rgr input
            # first channel is the foreground mask
            fg_mask = (self.render()[..., -1] / self.global_scale < 0.599/0.8).astype(np.float32)
            # second channel is the goal mask
            subgoal_mask = (subgoal < 0.5).astype(np.float32)
            particle_num = res_rgr.infer_param(fg_mask, subgoal_mask)
            print('particle_num: %d' % particle_num)
            # particle_r = np.sqrt(1.0 / particle_den)
            # particle_den_seq.append(particle_den)
            particle_den_seq.append(particle_num)
        # particle_den = np.array([1 / (particle_r * particle_r)])

        # return values
        rewards = np.zeros(n_mpc+1)
        raw_obs = np.zeros((n_mpc+1, self.screenHeight, self.screenWidth, 5))
        # states = np.zeros((n_mpc+1, particle_num, 3))
        states = []
        actions = np.zeros((n_mpc, self.act_dim))
        # states_pred = np.zeros((n_mpc, particle_num, 3))
        states_pred = []
        rew_means = np.zeros((n_mpc, 1, n_update_iter * gd_loop))
        rew_stds = np.zeros((n_mpc, 1, n_update_iter * gd_loop))

        if init_pos is not None:
            self.set_positions(init_pos)
        obs_cur = self.render()
        raw_obs[0] = obs_cur
        
        obs_cur, particle_r = self.obs2ptcl_fixed_num_batch(obs_cur, particle_num, batch_size=30)
        
        particle_den = np.array([1 / (particle_r * particle_r)])[0]
        print('particle_den:', particle_den)
        print('particle_num:', particle_num)
        # obs_cur = self.obs2ptcl(obs_cur, particle_r)
        if action_seq_mpc_init is None:
            action_seq_mpc_init, action_label_seq_mpc_init = self.sample_action(n_mpc)
        subgoal_tensor = torch.from_numpy(subgoal).float().cuda().reshape(self.screenHeight, self.screenWidth)
        subgoal_coor_tensor = torch.flip((subgoal_tensor < 0.5).nonzero(), dims=(1,)).cuda().float()
        subgoal_coor_np, _ = fps_np(subgoal_coor_tensor.detach().cpu().numpy(), min(particle_num * 5, subgoal_coor_tensor.shape[0]))
        subgoal_coor_tensor = torch.from_numpy(subgoal_coor_np).float().cuda()
        rewards[0] = config_reward_ptcl(torch.from_numpy(obs_cur).float().cuda().reshape(-1, particle_num, 3),
                                        subgoal_tensor,
                                        cam_params=self.get_cam_params(),
                                        goal_coor=subgoal_coor_tensor,
                                        normalize=True)[0].item()
        states.append(obs_cur[0])
        total_time = 0.0
        rollout_time = 0.0
        optim_time = 0.0
        iter_num = 0
        for i in range(n_mpc):
            attr_cur = np.zeros((obs_cur.shape[0], particle_num))
            init_p = self.get_positions()
            # print('mpc iter: {}'.format(i))
            # print('action_seq_mpc_init: {}'.format(action_seq_mpc_init.shape))
            # print('action_label_seq_mpc_init: {}'.format(action_label_seq_mpc_init))
            traj_opt_out = self.planner.trajectory_optimization_ptcl_multi_traj(
                            obs_cur,
                            particle_den,
                            attr_cur,
                            obs_goal=subgoal,
                            model_dy=model_dy,
                            act_seq=action_seq_mpc_init[:n_look_ahead],
                            act_label_seq=action_label_seq_mpc_init[:n_look_ahead] if action_label_seq_mpc_init is not None else None,
                            n_sample=n_sample,
                            n_look_ahead=min(n_look_ahead, n_mpc - i),
                            n_update_iter=n_update_iter,
                            action_lower_lim=action_lower_lim,
                            action_upper_lim=action_upper_lim,
                            use_gpu=True,
                            rollout_best_action_sequence=True,
                            reward_params=reward_params,
                            gd_loop=gd_loop,
                            time_lim=time_lim,)
            action_seq_mpc = traj_opt_out['action_sequence']
            next_r = traj_opt_out['next_r']
            obs_pred = traj_opt_out['observation_sequence'][0]
            rew_mean = traj_opt_out['rew_mean']
            rew_std = traj_opt_out['rew_std']
            iter_num += traj_opt_out['iter_num']
            
            print('mpc_step:', i)
            print('action:', action_seq_mpc[0])
            
            obs_cur = self.step(action_seq_mpc[0])
            if obs_cur is None:
                raise Exception('sim exploded')

            if auto_particle_r:
                # construct res_rgr input
                # first channel is the foreground mask
                fg_mask = (self.render()[..., -1] / self.global_scale < 0.599/0.8).astype(np.float32)
                # second channel is the goal mask
                subgoal_mask = (subgoal < 0.5).astype(np.float32)
                particle_num = res_rgr.infer_param(fg_mask, subgoal_mask)
                # particle_den = np.array([res_rgr(res_rgr_input[None, ...]).item() * 4000.0])
                # particle_r = np.sqrt(1.0 / particle_den)
                # particle_den_seq.append(particle_den)
                particle_den_seq.append(particle_num)

            raw_obs[i + 1] = obs_cur
            obs_cur, particle_r = self.obs2ptcl_fixed_num_batch(obs_cur, particle_num, batch_size=30)
            particle_den = np.array([1 / (particle_r ** 2)])[0]
            print('particle_den:', particle_den)
            print('particle_num:', particle_num)
            # obs_cur = self.obs2ptcl(obs_cur, particle_r)
            states.append(obs_cur[0])
            actions[i] = action_seq_mpc[0]
            subgoal_coor_np, _ = fps_np(subgoal_coor_tensor.detach().cpu().numpy(), min(obs_cur.shape[0] * 5, subgoal_coor_tensor.shape[0]))
            subgoal_coor_tensor = torch.from_numpy(subgoal_coor_np).float().cuda()
            rewards[i + 1] = config_reward_ptcl(torch.from_numpy(obs_cur).float().cuda().reshape(-1, particle_num, 3),
                                                subgoal_tensor,
                                                cam_params=self.get_cam_params(),
                                                goal_coor=subgoal_coor_tensor,
                                                normalize=True)[0].item()
            total_time += traj_opt_out['times']['total_time']
            rollout_time += traj_opt_out['times']['rollout_time']
            optim_time += traj_opt_out['times']['optim_time']
            states_pred.append(obs_pred)

            rew_means[i] = rew_mean
            rew_stds[i] = rew_std
            # action_seq_mpc_init = action_seq_mpc[1:]
            if action_seq_mpc_init.shape[0] > 1:
                action_seq_mpc_init = np.concatenate((traj_opt_out['action_full'][1:], action_seq_mpc_init[n_look_ahead:]), axis=0)
                if action_label_seq_mpc_init is not None:
                    action_label_seq_mpc_init = action_label_seq_mpc_init[1:]

            print('rewards: {}'.format(rewards))
            print()
        return {'rewards': rewards,
                'raw_obs': raw_obs,
                'states': states,
                'actions': actions,
                'states_pred': states_pred,
                'rew_means': rew_means,
                'rew_stds': rew_stds,
                'total_time': total_time,
                'rollout_time': rollout_time,
                'optim_time': optim_time,
                'iter_num': iter_num,
                'particle_den_seq': particle_den_seq,}
    
    def get_cam_params(self):
        # return fx, fy, cx, cy
        projMat = pyflex.get_projMatrix().reshape(4, 4).T
        cx = self.screenWidth / 2.0
        cy = self.screenHeight / 2.0
        fx = projMat[0, 0] * cx
        fy = projMat[1, 1] * cy
        return [fx, fy, cx, cy]
    
    def get_cam_extrinsics(self):
        return np.array(pyflex.get_viewMatrix()).reshape(4, 4).T
    
    def get_positions(self):
        return pyflex.get_positions()
    
    def set_positions(self, positions):
        pyflex.set_positions(positions)
        # for i in range(200):
        #     pyflex.step()

    def pixel2action(self, pixel, w = 64):
        x = (pixel[1] - w / 2) * 0.6 * self.global_scale / w
        y = (w / 2 - pixel[0]) * 0.6 * self.global_scale / w
        return np.array([x, y])

    def close(self):
        pyflex.clean()

