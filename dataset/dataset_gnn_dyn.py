import os
import numpy as np
import json
import cv2
import pdb
import pickle
import json
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from env.flex_env import FlexEnv

from scipy.spatial import KDTree
from utils import load_yaml, set_seed, fps_rad, fps_np, recenter, opengl2cam, depth2fgpcd, pcd2pix

import matplotlib.pyplot as plt
from dgl.geometry import farthest_point_sampler

from torch.utils.data import DataLoader

np.seterr(divide='ignore', invalid='ignore')

class ParticleDataset(Dataset):
    def __init__(self, data_dir, config, phase, cam):
        self.config = config

        n_episode = config['dataset']['n_episode']
        n_timestep = config['dataset']['n_timestep']
        self.global_scale = config['dataset']['global_scale']

        train_valid_ratio = config['train']['train_valid_ratio']

        n_train = int(n_episode * train_valid_ratio)
        n_valid = n_episode - n_train

        if phase == 'train':
            self.epi_st_idx = 0
            self.n_episode = n_train
        elif phase == 'valid':
            self.epi_st_idx = n_train
            self.n_episode = n_valid
        else:
            raise AssertionError("Unknown phase %s" % phase)

        self.n_timestep = n_timestep + 1
        self.n_his = config['train']['n_history']
        self.n_roll = config['train']['n_rollout']
        self.data_dir = data_dir

        self.screenHeight = 720
        self.screenWidth = 720
        self.img_channel = 1

        self.cam_params, self.cam_extrinsic = cam

    def __len__(self):
        return self.n_episode * (self.n_timestep - self.n_his - self.n_roll + 1)
    
    def read_particles(self, particles_path):
        particles = np.load(particles_path).reshape(-1, 4)
        particles[:, 3] = 1.0
        opencv_T_opengl = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        opencv_T_world = np.matmul(np.linalg.inv(self.cam_extrinsic), opencv_T_opengl)
        # print('opencv_T_world', opencv_T_world)
        # print('opencv_T_world inverse', np.linalg.inv(opencv_T_world))
        particles = np.matmul(np.linalg.inv(opencv_T_world), particles.T).T[:, :3] / self.global_scale
        return particles

    def __getitem__(self, idx):

        # particle_den_candidates = [200., 500., 1000., 2000.]
        particle_den_min = 15
        particle_den_max = 6500
        # particle_den_min = 6500
        # particle_den_max = 6500
        # particle_den = particle_den_candidates[np.random.randint(0, len(particle_den_candidates))]
        particle_den = np.random.uniform(particle_den_min, particle_den_max)
        particle_r = 1/np.sqrt(particle_den)

        offset = self.n_timestep - self.n_his - self.n_roll + 1
        idx_episode = idx // offset + self.epi_st_idx
        idx_timestep = idx % offset

        action_path = os.path.join(self.data_dir, '%d/actions.p' % idx_episode)
        with open(action_path, 'rb') as fp:
            actions = pickle.load(fp)

        # sample particles to track in the first frame
        first_depth_path = os.path.join(self.data_dir, '%d/%d_depth.png' % (idx_episode, idx_timestep))
        first_depth = cv2.imread(first_depth_path, cv2.IMREAD_ANYDEPTH) / (self.global_scale * 1000.0)
        first_depth_fgpcd = depth2fgpcd(first_depth, (first_depth < 0.599/0.8), self.cam_params) # [N, 3]
        sampled_pts = fps_rad(first_depth_fgpcd, particle_r) # [particle_num, 3]
        particle_num = sampled_pts.shape[0]
        sampled_pts = recenter(first_depth_fgpcd, sampled_pts, r = min(0.02, 0.5 * particle_r)) # [particle_num, 3]

        # find the nearest gt particle to sampled_pts
        first_particles_path = os.path.join(self.data_dir, '%d/%d_particles.npy' % (idx_episode, idx_timestep))
        first_particles = self.read_particles(first_particles_path)
        # print(first_particles)
        
        first_particles_tree = KDTree(first_particles)
        _, nearest_idx = first_particles_tree.query(sampled_pts, k=1)
        
        states = np.zeros((self.n_his + self.n_roll, particle_num, 3))
        color_imgs = np.zeros((self.n_his + self.n_roll, 720, 720, 3)).astype(np.uint8)
        states_delta = np.zeros((self.n_his + self.n_roll - 1, particle_num, 3))
        attrs = np.zeros(states.shape[:2])

        for i in range(idx_timestep, idx_timestep + self.n_his + self.n_roll):
            particles_path = os.path.join(self.data_dir, '%d/%d_particles.npy' % (idx_episode, i))
            particles = self.read_particles(particles_path)
            states[i - idx_timestep] = particles[nearest_idx, :]

            if i < idx_timestep + self.n_his + self.n_roll - 1:
                s = actions[i, :2]
                e = actions[i, 2:]
                h = 0.0
                pusher_w = 0.8 / 24.0

                # # construct grids for testing
                # grid_2d = np.mgrid[0:720, 0:720].transpose(2, 1, 0)
                # grid_2d = grid_2d.reshape(-1, 2)
                # fx, fy, cx, cy = self.cam_params
                # grid_3d = np.zeros((720 * 720, 3))
                # grid_3d[:, 0] = (grid_2d[:, 0] - cx) * 0.75 / fx
                # grid_3d[:, 1] = (grid_2d[:, 1] - cy) * 0.75 / fy
                # grid_3d[:, 2] = 0.75

                s_3d = np.array([s[0], h, -s[1]])
                e_3d = np.array([e[0], h, -e[1]])
                s_3d_cam = opengl2cam(s_3d[None, :], self.cam_extrinsic, self.global_scale)[0]
                e_3d_cam = opengl2cam(e_3d[None, :], self.cam_extrinsic, self.global_scale)[0]
                push_dir_cam = e_3d_cam - s_3d_cam
                push_l = np.linalg.norm(push_dir_cam)
                push_dir_cam = push_dir_cam / np.linalg.norm(push_dir_cam)
                try:
                    assert abs(push_dir_cam[2]) < 1e-6
                except:
                    print(push_dir_cam)
                    exit(1)
                push_dir_ortho_cam = np.array([-push_dir_cam[1], push_dir_cam[0], 0.0])
                pos_diff_cam = particles[nearest_idx, :] - s_3d_cam[None, :] # [particle_num, 3]
                pos_diff_ortho_proj_cam = (pos_diff_cam * np.tile(push_dir_ortho_cam[None, :], (particle_num, 1))).sum(axis=1) # [particle_num,]
                pos_diff_proj_cam = (pos_diff_cam * np.tile(push_dir_cam[None, :], (particle_num, 1))).sum(axis=1) # [particle_num,]
                # pos_diff_cam_grid_test = grid_3d - s_3d_cam[None, :] # [720*720, 3]
                # pos_diff_ortho_proj_cam_grid_test = (pos_diff_cam_grid_test * np.tile(push_dir_ortho_cam[None, :], (720 * 720, 1))).sum(axis=1) # [720*720,]
                # pos_diff_proj_cam_grid_test = (pos_diff_cam_grid_test * np.tile(push_dir_cam[None, :], (720 * 720, 1))).sum(axis=1) # [720*720,]
                # pos_diff_l_mask_grid_test = ((pos_diff_proj_cam_grid_test < push_l) & (pos_diff_proj_cam_grid_test > 0.0)).astype(np.float32) # hard mask
                # pos_diff_w_mask_grid_test = np.maximum(np.maximum(-pusher_w - pos_diff_ortho_proj_cam_grid_test, 0.), # soft mask
                #                             np.maximum(pos_diff_ortho_proj_cam_grid_test - pusher_w, 0.))
                # pos_diff_w_mask_grid_test = np.exp(-pos_diff_w_mask_grid_test / 0.01) # [particle_num,]

                # push_dir = e - s
                # push_theta = np.arctan2(push_dir[1], push_dir[0])
                # pusher_s = np.zeros((3, 3))
                # pusher_s[0, :] = s_3d - 24. * pusher_w * np.array([np.sin(push_theta), 0.0, np.cos(push_theta)])
                # pusher_s[1, :] = s_3d
                # pusher_s[2, :] = s_3d + 24. * pusher_w * np.array([np.sin(push_theta), 0.0, np.cos(push_theta)])
                # pusher_s = self.flex2cam(pusher_s)
                # pusher_s_2d = self.ptcl2pix(pusher_s)
                # pusher_e = np.zeros((3, 3))
                # pusher_e[0, :] = e_3d - 24. * pusher_w * np.array([np.sin(push_theta), 0.0, np.cos(push_theta)])
                # pusher_e[1, :] = e_3d
                # pusher_e[2, :] = e_3d + 24. * pusher_w * np.array([np.sin(push_theta), 0.0, np.cos(push_theta)])
                # pusher_e = self.flex2cam(pusher_e)
                # pusher_e_2d = self.ptcl2pix(pusher_e)

                # ref_img = np.ones((720, 720, 3)) * 255
                # for j in range(3):
                #     ref_img = cv2.circle(ref_img.copy(), (int(pusher_s_2d[j, 0]), int(pusher_s_2d[j, 1])), 5, (255, 0, 0), -1)
                # for j in range(3):
                #     ref_img = cv2.circle(ref_img.copy(), (int(pusher_e_2d[j, 0]), int(pusher_e_2d[j, 1])), 5, (0, 255, 0), -1)
                # ref_img = ref_img.astype(np.uint8)
                # plt.subplot(1, 3, 1)
                # plt.imshow(pos_diff_l_mask_grid_test.reshape(720, 720))
                # plt.subplot(1, 3, 2)
                # plt.imshow(pos_diff_w_mask_grid_test.reshape(720, 720))
                # plt.subplot(1, 3, 3)
                # plt.imshow(ref_img)
                # plt.show()
                pos_diff_l_mask = ((pos_diff_proj_cam < push_l) & (pos_diff_proj_cam > 0.0)).astype(np.float32) # hard mask
                pos_diff_w_mask = np.maximum(np.maximum(-pusher_w - pos_diff_ortho_proj_cam, 0.), # soft mask
                                            np.maximum(pos_diff_ortho_proj_cam - pusher_w, 0.))
                pos_diff_w_mask = np.exp(-pos_diff_w_mask / 0.01) # [particle_num,]
                pos_diff_to_end_cam = (e_3d_cam[None, :] - particles[nearest_idx, :]) # [particle_num, 3]
                pos_diff_to_end_cam = (pos_diff_to_end_cam * np.tile(push_dir_cam[None, :], (particle_num, 1))).sum(axis=1) # [particle_num,]
                states_delta[i - idx_timestep] = pos_diff_to_end_cam[:, None] * push_dir_cam[None, :] * pos_diff_l_mask[:, None] * pos_diff_w_mask[:, None]
            color_img_path = os.path.join(self.data_dir, '%d/%d_color.png' % (idx_episode, i))
            color_img = cv2.imread(color_img_path)
            color_imgs[i - idx_timestep, :, :, :] = color_img
        states = torch.FloatTensor(states)
        states_delta = torch.FloatTensor(states_delta)
        attrs = torch.FloatTensor(attrs)
        return states, states_delta, attrs, particle_num, particle_den, color_imgs

def dataset_test():
    config = load_yaml('config.yaml')

    cam = []
    env = FlexEnv(config)
    env.reset()
    cam.append(env.get_cam_params())
    cam.append(env.get_cam_extrinsics())
    env.close()

    dataset = ParticleDataset(config['train']['data_root'], config, 'train', cam)
    states, states_delta, attrs, particle_num, color_imgs = dataset[0]
    vid = cv2.VideoWriter('dataset_nopusher.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (720, 720))
    for i in range(states.shape[0] - 1):
        img = color_imgs[i]
        obj_pix = pcd2pix(states[i], cam[0])
        next_pix = pcd2pix(states[i] + states_delta[i], cam[0])
        for j in range(obj_pix.shape[0]):
            img = cv2.circle(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])), 5, (0, 0, 255), -1)
        for j in range(next_pix.shape[0]):
            img = cv2.arrowedLine(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])),
                                  (int(next_pix[j, 1]), int(next_pix[j, 0])), (0, 255, 0), 2)
        vid.write(img)
    vid.release()

def calibrate_res_range():
    config = load_yaml('config.yaml')
    env = FlexEnv(config)
    
    env.init_pos = 'rb_corner'
    env.reset()

    raw_obs = env.render()
    depth = raw_obs[..., -1] / config['dataset']['global_scale']

    depth_fgpcd = depth2fgpcd(depth, (depth < 0.599/0.8), env.get_cam_params()) # [N, 3]
    sampled_pts, min_particle_r = fps_np(depth_fgpcd, 100)
    max_particle_den = 1 / (min_particle_r ** 2)
    print('max_particle_den: %f' % max_particle_den)
    
    env.init_pos = 'extra_small_wkspc_spread'
    env.reset()
    
    raw_obs = env.render()
    depth = raw_obs[..., -1] / config['dataset']['global_scale']
    depth_fgpcd = depth2fgpcd(depth, (depth < 0.599/0.8), env.get_cam_params()) # [N, 3]
    sampled_pts, max_particle_r = fps_np(depth_fgpcd, 2)
    min_particle_den = 1 / (max_particle_r ** 2)
    print('min_particle_den: %f' % min_particle_den)
    

if __name__ == '__main__':
    # dataset_test()
    calibrate_res_range()
