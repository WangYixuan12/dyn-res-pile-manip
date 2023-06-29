from distutils.command.config import config
import os
import torch
import time
import numpy as np
import scipy.stats as stats

import torch.optim as optim

from scipy.special import softmax

# netcompdy
from model.gnn_dyn import PropNetDiffDenModel
from env.flex_rewards import highest_reward, config_reward, distractor_reward, distractor_reward_diff, config_reward_ptcl

from utils import fps_np, pcd2pix

import matplotlib.pyplot as plt

DEBUG = False

# workaround for np.cross leads to unreachable code; see https://github.com/microsoft/pylance-release/issues/3277
cross = lambda x,y:np.cross(x,y)

def particle_num_to_iter_time(particle_num):
    # fitted using https://keisan.casio.com/exec/system/14059932254941
    time_bound_iter = (2969.3971 - 69.923244 * particle_num + 1.8509846 * particle_num ** 2) / 200. # batch size 300
    return max(int(time_bound_iter), 1)

class Planner(object):

    def __init__(self, config, env):
        self.config = config
        self.action_dim = 4
        self.global_scale = config['dataset']['global_scale']
        self.img_ch = 1
        self.n_his = config['train']['n_history']
        self.env = env

        self.cam_params = self.env.get_cam_params()
        self.is_real = self.env.is_real
        if not self.is_real:
            self.cam_extrinsic = self.env.get_cam_extrinsics()
        self.screenHeight = self.env.screenHeight
        self.screenWidth = self.env.screenWidth

    def evaluate_traj(self, obs_seqs, obs_goal):
        # obs_seqs: [n_sample, n_look_ahead, state_dim]
        # obs_goal: state_dim
        pass

    def optimize_action(self, act_seqs, reward_seqs):
        pass

    def trajectory_optimization(self,
                                state_cur,  # current state, shape: [n_his, state_dim]
                                obs_goal,  # goal, shape: [state_dim]
                                model_dy,  # the learned dynamics model
                                act_seq,  # initial action sequence, shape: [-1, action_dim]
                                n_sample, n_look_ahead, n_update_iter,
                                action_lower_lim, action_upper_lim, use_gpu):
        pass

class PlannerGD(Planner):

    def __init__(self, config, env):
        super(PlannerGD, self).__init__(config, env)

    def sample_action_sequences(self,
                                init_act_seq,  # unnormalized, shape: [n_his + n_look_ahead - 1, action_dim] / [n_his + n_look_ahead - 1, traj_num, action_dim]
                                init_act_label_seq,  # integer, shape: [n_his + n_look_ahead - 1]
                                n_sample,  # integer, number of action trajectories to sample
                                action_lower_lim,  # unnormalized, shape: action_dim, lower limit of the action
                                action_upper_lim,  # unnormalized, shape: action_dim, upper limit of the action
                                noise_type="normal"):
        init_act_seq_dim = len(init_act_seq.shape)
        if DEBUG:
            # Input check begin
            print('-----------------')
            print('check input for sample_action_sequences')
            assert init_act_seq_dim == 2 or init_act_seq_dim == 3
            assert type(init_act_seq) == np.ndarray
            print('init_act_seq.shape', init_act_seq.shape)
            # print("init_act_seq", init_act_seq) # HEAVY
            if init_act_seq_dim == 2:
                assert type(init_act_label_seq) == np.ndarray
                print('init_act_label_seq.shape', init_act_label_seq.shape)
                # print('init_act_label_seq', init_act_label_seq)
            print('n_sample', n_sample)
            print()
            # Input check end

        beta_filter = self.config['mpc']['mppi']['beta_filter']
        if init_act_seq_dim == 3:
            n_look_ahead, traj_num, action_dim = init_act_seq.shape
        elif init_act_seq_dim == 2:
            n_look_ahead, action_dim = init_act_seq.shape

        # [n_sample, -1, action_dim] / [n_sample, -1, traj_num, action_dim]
        act_seqs = np.stack([init_act_seq] * n_sample)

        # [n_sample, action_dim] / [n_sample, traj_num, action_dim]
        if init_act_seq_dim == 2:
            act_residual = np.zeros((n_sample, self.action_dim))
        elif init_act_seq_dim == 3:
            act_residual = np.zeros((n_sample, traj_num, self.action_dim))

        # only add noise to future actions init_act_seq[:(n_his-1)] are past
        # The action we are optimizing for the current timestep is in fact
        # act_seq[n_his - 1].

        # actions that go as input to the dynamics network
        for i in range(self.n_his-1, init_act_seq.shape[0]):

            if noise_type == "normal":
                sigma = self.config['mpc']['sigma'] * self.global_scale / 12.0

                # [n_sample, action_dim]
                if init_act_seq_dim == 2:
                    noise_sample = np.random.normal(0, sigma, (n_sample, self.action_dim))
                elif init_act_seq_dim == 3:
                    noise_sample = np.random.normal(0, sigma, (n_sample, traj_num, self.action_dim))
            elif noise_type == "uniform":
                sigma = 2.0 * self.global_scale / 12.0

                # [n_sample, action_dim]
                if init_act_seq_dim == 2:
                    noise_sample = np.random.uniform(-sigma, sigma, (n_sample, self.action_dim))
                elif init_act_seq_dim == 3:
                    noise_sample = np.random.uniform(-sigma, sigma, (n_sample, traj_num, self.action_dim))
            elif noise_type == "total_rand":
                if init_act_seq_dim == 2:
                    noise_sample = np.zeros((n_sample, self.action_dim))
                elif init_act_seq_dim == 3:
                    noise_sample = np.zeros((n_sample, traj_num, self.action_dim))
            else:
                raise ValueError("unknown noise type: %s" %(noise_type))

            # print("noise.shape", noise.shape)
            # noise = u_t in MPPI paper

            # act_residual = n_t in MPPI paper
            # should we clip act_residual also . . . , probably not since it is zero centered
            act_residual = beta_filter * noise_sample + act_residual * (1. - beta_filter)

            # add the perturbation to the action sequence
            act_seqs[:, i] += act_residual

            # clip to range
            if init_act_seq_dim == 2:
                cvx_l = int(init_act_label_seq[i])
                x_diff = self.env.cvx_region[cvx_l, 1] - self.env.cvx_region[cvx_l, 0]
                y_diff = self.env.cvx_region[cvx_l, 3] - self.env.cvx_region[cvx_l, 2]
                cvx_lower_lim = np.array([self.env.cvx_region[cvx_l, 0], self.env.cvx_region[cvx_l, 2], self.env.cvx_region[cvx_l, 0] + x_diff * 0.15, self.env.cvx_region[cvx_l, 2] + y_diff * 0.15])
                cvx_upper_lim = np.array([self.env.cvx_region[cvx_l, 1], self.env.cvx_region[cvx_l, 3], self.env.cvx_region[cvx_l, 1] - x_diff * 0.15, self.env.cvx_region[cvx_l, 3] - y_diff * 0.15])
                # print('cvx_lower_lim', cvx_lower_lim)
                # print('cvx_upper_lim', cvx_upper_lim)
                # input()
                act_seqs[:, i] = np.clip(act_seqs[:, i], cvx_lower_lim, cvx_upper_lim)
                # act_seqs[:, i] = np.clip(act_seqs[:, i], action_lower_lim, action_upper_lim)
            elif init_act_seq_dim == 3:
                for cvx_l in range(1):
                    x_diff = self.env.cvx_region[cvx_l, 1] - self.env.cvx_region[cvx_l, 0]
                    y_diff = self.env.cvx_region[cvx_l, 3] - self.env.cvx_region[cvx_l, 2]
                    cvx_lower_lim = np.array([self.env.cvx_region[cvx_l, 0], self.env.cvx_region[cvx_l, 2], self.env.cvx_region[cvx_l, 0] + x_diff * 0.15, self.env.cvx_region[cvx_l, 2] + y_diff * 0.15])
                    cvx_upper_lim = np.array([self.env.cvx_region[cvx_l, 1], self.env.cvx_region[cvx_l, 3], self.env.cvx_region[cvx_l, 1] - x_diff * 0.15, self.env.cvx_region[cvx_l, 3] - y_diff * 0.15])
                    act_seqs[:, i, cvx_l] = np.clip(act_seqs[:, i, cvx_l], cvx_lower_lim, cvx_upper_lim)
            
            if noise_type == 'total_rand':
                for cvx_l in range(1):
                    x_diff = self.env.cvx_region[cvx_l, 1] - self.env.cvx_region[cvx_l, 0]
                    y_diff = self.env.cvx_region[cvx_l, 3] - self.env.cvx_region[cvx_l, 2]
                    cvx_lower_lim = np.array([self.env.cvx_region[cvx_l, 0], self.env.cvx_region[cvx_l, 2], self.env.cvx_region[cvx_l, 0] + x_diff * 0.15, self.env.cvx_region[cvx_l, 2] + y_diff * 0.15])
                    cvx_upper_lim = np.array([self.env.cvx_region[cvx_l, 1], self.env.cvx_region[cvx_l, 3], self.env.cvx_region[cvx_l, 1] - x_diff * 0.15, self.env.cvx_region[cvx_l, 3] - y_diff * 0.15])
                    act_seqs[:, i, cvx_l] = np.random.uniform(cvx_lower_lim, cvx_upper_lim, (n_sample, self.action_dim))

        if DEBUG:
            # Output check begin
            # act_seqs: [n_sample, -1, action_dim]
            print("check output for sample_action_sequences")
            print("act_seqs.shape", act_seqs.shape)
            # print("act_seqs", act_seqs) # HEAVY
            assert act_seqs.shape[0] == n_sample
            assert act_seqs.shape[1] == init_act_seq.shape[0]
            assert act_seqs.shape[2] == self.action_dim
            print('-----------------')
            print()
            # Output check end

        return act_seqs

    def world2cam(self, world_pts):
        # world_pts: (n, 3) torch Tensor
        # cam: (n, 3)
        # print('flex', flex)
        assert type(world_pts) == torch.Tensor
        opencv_T_opengl = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        opencv_T_world = np.matmul(np.linalg.inv(self.cam_extrinsic), opencv_T_opengl)
        opencv_T_world_inv = np.linalg.inv(opencv_T_world)
        opencv_T_world_inv_tensor = torch.tensor(opencv_T_world_inv, device=world_pts.device).float()
        dummy_one = torch.ones((world_pts.shape[0], 1), device=world_pts.device, dtype=world_pts.dtype)
        # print('opencv_T_world inverse', np.linalg.inv(opencv_T_world))
        cam = torch.matmul(opencv_T_world_inv_tensor, torch.concat([world_pts, dummy_one], dim=1).T).T[:, :3] / self.global_scale
        # print('cam', cam)
        # print()
        return cam

    def gen_s_delta(self, s_cur : torch.Tensor, action : torch.Tensor):
        # s_cur: (N, particle_num, 3) tensor
        # action: (N, 4) tensor
        # s_delta: (N, particle_num, 3) tensor
        assert type(s_cur) == torch.Tensor
        assert s_cur.shape[1:] == (self.particle_num, 3)
        assert s_cur.shape[0] == action.shape[0]

        assert type(action) == torch.Tensor
        # print('action[0]:', action[0])
        # print('s_cur[0]:', s_cur[0])
        N = action.shape[0]
        s = action[:, :2] # (N, 2)
        e = action[:, 2:] # (N, 2)
        h = 0.0 * torch.ones((s.shape[0], 1),
                              device=action.device,
                              dtype=action.dtype)
        pusher_w = 0.8 / 24.0

        
        s_3d = torch.concat([s[:, 0:1], h, -s[:, 1:2]], axis=1) # (N, 3)
        e_3d = torch.concat([e[:, 0:1], h, -e[:, 1:2]], axis=1) # (N, 3)
        s_3d_cam = self.world2cam(s_3d) # (N, 3)
        e_3d_cam = self.world2cam(e_3d) # (N, 3)
        # print('s_3d_cam[0]:', s_3d_cam[0])
        # print('e_3d_cam[0]:', e_3d_cam[0])

        push_dir_cam = e_3d_cam - s_3d_cam # (N, 3)
        push_l = torch.linalg.norm(push_dir_cam, axis = 1) # (N,)
        push_dir_cam = push_dir_cam / torch.linalg.norm(push_dir_cam, dim=1, keepdim=True) # (N, 3)
        dummy_zeros = torch.zeros((N, 1), device=push_dir_cam.device, dtype=push_dir_cam.dtype)
        push_dir_ortho_cam = torch.concat([-push_dir_cam[:, 1:2], push_dir_cam[:, 0:1], dummy_zeros], dim=1) # (N, 3)
        # z_unit = torch.Tensor([0.0, 0.0, 1.0]).to(device=action.device, dtype=action.dtype)
        # push_dir_ortho_cam = torch.cross(push_dir_cam, z_unit) # (N, 3)
        pos_diff_cam = s_cur - s_3d_cam[:, None, :] # [N, particle_num, 3]
        pos_diff_ortho_proj_cam = (pos_diff_cam * torch.tile(push_dir_ortho_cam[:, None, :], (1, self.particle_num, 1))).sum(axis=-1) # [N, particle_num,]
        pos_diff_proj_cam = (pos_diff_cam * torch.tile(push_dir_cam[:, None, :], (1, self.particle_num, 1))).sum(axis=-1) # [N, particle_num,]
        pos_diff_l_mask = ((pos_diff_proj_cam < push_l[:, None]) & (pos_diff_proj_cam > 0.0)).to(dtype = torch.float32) # hard mask [N, particle_num,]
        pos_diff_w_mask = torch.maximum(torch.clamp(-pusher_w - pos_diff_ortho_proj_cam, min=0.), # soft mask
                                    torch.clamp(pos_diff_ortho_proj_cam - pusher_w, min=0.))
        pos_diff_w_mask = torch.exp(-pos_diff_w_mask / 0.01) # [N, particle_num,]
        pos_diff_to_end_cam = (e_3d_cam[:, None, :] - s_cur) # [N, particle_num, 3]
        pos_diff_to_end_cam = (pos_diff_to_end_cam * torch.tile(push_dir_cam[:, None, :], (1, self.particle_num, 1))).sum(axis=-1) # [N, particle_num,]
        s_delta = pos_diff_to_end_cam[..., None] * push_dir_cam[:, None, :] * pos_diff_l_mask[..., None] * pos_diff_w_mask[..., None]
        # print(s_delta[0])
        assert s_delta.shape == (N, self.particle_num, 3)
        return s_delta

    def gen_s_delta_irl(self, s_cur : torch.Tensor, action : torch.Tensor):
        # s_cur: (N, particle_num, 3) tensor
        # action: (N, 4) tensor
        # s_delta: (N, particle_num, 3) tensor
        assert type(s_cur) == torch.Tensor
        assert s_cur.shape[1:] == (self.particle_num, 3)
        assert s_cur.shape[0] == action.shape[0]

        assert type(action) == torch.Tensor
        # print('action[0]:', action[0])
        # print('s_cur[0]:', s_cur[0])
        s_cur_shifted = s_cur.clone()
        s_cur_shifted[:, :, 0] -= self.env.wkspc_center_x
        s_cur_shifted[:, :, 1] -= self.env.wkspc_center_y
        N = action.shape[0]
        s = action[:, :2]
        e = action[:, 2:]
        h = 0.88 * torch.ones((s.shape[0], 1), device=action.device, dtype=action.dtype)
        s_3d_cam = torch.concat([s[:, 0:1] / self.env.s2r_scale, - s[:, 1:2] / self.env.s2r_scale, h], axis=1)
        e_3d_cam = torch.concat([e[:, 0:1] / self.env.s2r_scale, - e[:, 1:2] / self.env.s2r_scale, h], axis=1)
        pusher_w = 0.048

        push_dir_cam = e_3d_cam - s_3d_cam # (N, 3)
        push_l = torch.linalg.norm(push_dir_cam, axis = 1) # (N,)
        push_dir_cam = push_dir_cam / torch.linalg.norm(push_dir_cam, dim=1, keepdim=True) # (N, 3)
        dummy_zeros = torch.zeros((N, 1), device=push_dir_cam.device, dtype=push_dir_cam.dtype)
        push_dir_ortho_cam = torch.concat([-push_dir_cam[:, 1:2], push_dir_cam[:, 0:1], dummy_zeros], dim=1) # (N, 3)
        # z_unit = torch.Tensor([0.0, 0.0, 1.0]).to(device=action.device, dtype=action.dtype)
        # push_dir_ortho_cam = torch.cross(push_dir_cam, z_unit) # (N, 3)
        pos_diff_cam = s_cur_shifted - s_3d_cam[:, None, :] # [N, particle_num, 3]
        pos_diff_ortho_proj_cam = (pos_diff_cam * torch.tile(push_dir_ortho_cam[:, None, :], (1, self.particle_num, 1))).sum(axis=-1) # [N, particle_num,]
        pos_diff_proj_cam = (pos_diff_cam * torch.tile(push_dir_cam[:, None, :], (1, self.particle_num, 1))).sum(axis=-1) # [N, particle_num,]
        pos_diff_l_mask = ((pos_diff_proj_cam < push_l[:, None]) & (pos_diff_proj_cam > 0.0)).to(dtype = torch.float32) # hard mask [N, particle_num,]
        pos_diff_w_mask = torch.maximum(torch.clamp(-pusher_w - pos_diff_ortho_proj_cam, min=0.), # soft mask
                                    torch.clamp(pos_diff_ortho_proj_cam - pusher_w, min=0.))
        pos_diff_w_mask = torch.exp(-pos_diff_w_mask / 0.01) # [N, particle_num,]
        pos_diff_to_end_cam = (e_3d_cam[:, None, :] - s_cur_shifted) # [N, particle_num, 3]
        pos_diff_to_end_cam = (pos_diff_to_end_cam * torch.tile(push_dir_cam[:, None, :], (1, self.particle_num, 1))).sum(axis=-1) # [N, particle_num,]
        s_delta = pos_diff_to_end_cam[..., None] * push_dir_cam[:, None, :] * pos_diff_l_mask[..., None] * pos_diff_w_mask[..., None]
        # print(s_delta[0])
        assert s_delta.shape == (N, self.particle_num, 3)
        return s_delta

    def ptcl_model_rollout(self,
                           s_cur_tensor,  # the current state, shape: [n_batch, particle_num, 3]
                           s_param_tensor, # the parameter of dynamics model, shape: [n_batch,]
                           a_cur_tensor, # the current attributes, shape: [n_batch, particle_num]
                           model_dy,  # the learned dynamics model
                           act_seqs,  # the sampled action sequences, pytorch tensor, unnormalized, shape: [n_sample * n_batch, -1, action_dim]
                           enable_grad = True):
        n_sample_times_n_batch, N, action_dim = act_seqs.size()
        n_batch = s_cur_tensor.shape[0]
        n_sample = n_sample_times_n_batch // n_batch
        assert type(s_cur_tensor) == torch.Tensor
        assert type(a_cur_tensor) == torch.Tensor
        # assert s_cur_tensor.shape[0] == self.n_his
        assert s_cur_tensor.shape[1] == self.particle_num
        assert s_cur_tensor.shape[2] == 3
        # assert a_cur_tensor.shape[0] == self.n_his
        assert a_cur_tensor.shape[1] == self.particle_num
        assert type(act_seqs) == torch.Tensor
        if DEBUG:
            # Input check begin
            print('-----------------')
            print("check input for model_rollout")
            print("state_cur_np.shape", s_cur_tensor.shape)
            # viz the current state
            # print('viz the current state')
            # plt.subplot(1, 2, 1)
            # plt.imshow(state_cur_np[0].reshape(4, 64, 64)[3])
            # plt.subplot(1, 2, 2)
            # plt.imshow(state_cur_np[0].reshape(4, 64, 64)[:3].transpose(1, 2, 0))
            # plt.show()
            print("act_seqs_tensor.shape", act_seqs.shape)
            print()
            # Input check end

        states_pred_tensor = torch.zeros((n_sample * n_batch, N, self.particle_num, 3)).to(device = s_cur_tensor.device, dtype = s_cur_tensor.dtype)
        s_cur_tensor = torch.tile(s_cur_tensor, (n_sample, 1, 1)) # (n_sample * n_batch, particle_num, 3)
        s_param_tensor = torch.tile(s_param_tensor, (n_sample, )) # (n_sample * n_batch)
        a_cur_tensor = torch.tile(a_cur_tensor, (n_sample, 1)) # (n_sample * n_batch, particle_num)
        # act_seqs = act_seqs.repeat_interleave(n_batch, dim=0) # (n_sample * n_batch, N, action_dim)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rollout_time = 0.0
        for i in range(N):
            if not self.is_real:
                s_delta_tensor = self.gen_s_delta(s_cur_tensor, act_seqs[:, i, :]) # (n_sample * n_batch, particle_num, 3)
            else:
                s_delta_tensor = self.gen_s_delta_irl(s_cur_tensor, act_seqs[:, i, :]) # (n_sample * n_batch, particle_num, 3)
            
            start.record()
            # add condition
            if type(model_dy) == PropNetDiffDenModel:
                states_pred_tensor[:, i, :, :] = model_dy.predict_one_step(a_cur_tensor, s_cur_tensor, s_delta_tensor, s_param_tensor)
            else:
                raise NotImplementedError
            end.record()
            torch.cuda.synchronize()
            rollout_time += start.elapsed_time(end)
            s_cur_tensor = states_pred_tensor[:, i, :, :]

        out = {'state_pred': states_pred_tensor}

        if DEBUG:
            print("check output for model_rollout")
            print('state_pred.shape', out['state_pred'].shape)
            print('-----------------')
            print()
            # Output check end

        return {'model_rollout': out, 'rollout_time': rollout_time}

    def ptcl_evaluate_traj(self,
                           obs_seqs,
                           obs_goal,
                           obs_goal_coor_tensor,
                           debug=False,
                           funnel_dist=None,
                           distractor_df_fn=None,
                           act_seqs_tensor=None,
                           normalize_rew=True,):
        """
        Computes the reward as negative of l2 distance between obs_seqs[:, -1] and goal
        Input:
            obs_seqs: [n_sample, n_look_ahead, cvx_num, particle_num, 3] torch tensor
            obs_goal: [H, W] torch tensor
        Outpur:
            reward_seqs: [n_sample, cvx_num] torch tensor
            next_r: [n_sample, n_look_ahead, cvx_num] torch tensor
        """
        assert type(obs_seqs) == torch.Tensor
        assert len(obs_seqs.shape) == 5
        assert obs_seqs.shape[3] == self.particle_num
        assert obs_seqs.shape[4] == 3

        assert type(obs_goal) == torch.Tensor
        assert len(obs_goal.shape) == 2
        assert obs_goal.shape[0] == self.screenHeight
        assert obs_goal.shape[1] == self.screenWidth

        if DEBUG:
            # Input check begin
            print('-----------------')
            print("check input for evaluate_traj")
            print("obs_seqs.shape", obs_seqs.shape)
            print('obs_goal', obs_goal)
            print()
            # Input check end

        if self.env.is_real:
            offset = (- self.env.crop_w_lower + self.env.crop_w_off, - self.env.crop_h_lower + self.env.crop_h_off)
        else:
            offset = (0, 0)
        n_sample, n_look_ahead, cvx_num, _, _ = obs_seqs.shape
        obs_future = obs_seqs.reshape(n_sample * n_look_ahead * cvx_num, self.particle_num, 3)
        distractor_rew = torch.zeros(n_sample * n_look_ahead * cvx_num, device=obs_seqs.device, dtype=obs_seqs.dtype)
        if distractor_df_fn is None:
            next_r = config_reward_ptcl(obs_future,
                                        obs_goal,
                                        cam_params=self.cam_params,
                                        goal_coor=obs_goal_coor_tensor,
                                        normalize=normalize_rew,
                                        offset=offset,)
            # next_r: [n_sample * n_look_ahead * cvx_num]
        else:
            next_r = config_reward_ptcl(obs_future,
                                        obs_goal,
                                        cam_params=self.cam_params,
                                        goal_coor=obs_goal_coor_tensor,
                                        normalize=normalize_rew,
                                        offset=offset,)
            distractor_rew = distractor_reward_diff(act_seqs_tensor=act_seqs_tensor,
                                                    distractor_dist_fn=distractor_df_fn,
                                                    config=self.config,
                                                    debug=debug,
                                                    width=self.screenWidth,)
        next_r = next_r.reshape(n_sample, n_look_ahead, cvx_num)
        distractor_rew = distractor_rew.reshape(n_sample, n_look_ahead, cvx_num)
        reward_seqs = next_r[:, -1] + distractor_rew.sum(axis = 1)

        # reward_seqs: n_sample
        # next_r: n_sample, n_look_ahead
        if DEBUG:
            # Output check begin
            print("check output for evaluate_traj")
            print("reward_seqs.shape", reward_seqs.shape)
            assert type(reward_seqs) == torch.Tensor
            print("next_r.shape", next_r.shape)
            assert type(next_r) == torch.Tensor
            print('-----------------')
        assert reward_seqs.shape == (n_sample, cvx_num)
        assert next_r.shape == (n_sample, n_look_ahead, cvx_num)
        return reward_seqs, next_r

    def evaluate_traj(self,
                      obs_seqs,
                      obs_goal,
                      obs_goal_mask_tensor,
                      debug=False,
                      distractor_df_fn=None,
                      act_seqs_tensor=None):
        """
        Computes the reward as negative of l2 distance between obs_seqs[:, -1] and goal
        :param obs_seqs: [n_sample, n_look_ahead, state_dim] / [n_sample, n_look_ahead, cvx_num, state_dim] torch tensor
        :type obs_seqs: np.ndarray
        :param obs_goal: 
        :type obs_goal:
        :return: (reward_seqs, next_r) where reward_seqs is [n_sample] / [n_sample, cvx_num] and next_r is [n_sample, n_look_ahead] / [n_sample, n_look_ahead, cvx_num]
        :rtype:
        """
        assert type(obs_seqs) == torch.Tensor
        assert len(obs_seqs.shape) == 4
        # assert obs_seqs.shape[3] == self.state_dim

        assert type(obs_goal) == torch.Tensor
        assert len(obs_goal.shape) == 2
        # assert obs_goal.shape == (self.state_h, self.state_w)

        assert type(obs_goal_mask_tensor) == torch.Tensor
        assert len(obs_goal_mask_tensor.shape) == 2
        # assert obs_goal_mask_tensor.shape == (self.state_h, self.state_w)
        if DEBUG:
            # Input check begin
            print('-----------------')
            print("check input for evaluate_traj")
            print("obs_seqs.shape", obs_seqs.shape)
            print('obs_goal', obs_goal)
            print()
            # Input check end

        n_sample, n_look_ahead, cvx_num, _ = obs_seqs.shape
        res = int(np.sqrt(obs_seqs.shape[3]))
        obs_future = obs_seqs.reshape(n_sample * n_look_ahead * cvx_num, self.img_ch, res, res)
        # obs_final = obs_seqs[:, -1].reshape(n_sample, self.img_ch, self.state_h, self.state_w)
        if distractor_df_fn is None:
            next_r = config_reward(obs_future,
                                   obs_goal,
                                   obs_goal_mask_tensor,
                                   img_format='binary',)
            next_r = next_r.reshape(n_sample, n_look_ahead, cvx_num)
            reward_seqs = next_r[:, -1]
        else:
            next_r = config_reward(obs_future,
                                   obs_goal,
                                   obs_goal_mask_tensor,
                                   img_format='binary',)
            distractor_rew = distractor_reward_diff(act_seqs_tensor=act_seqs_tensor,
                                                distractor_dist_fn=distractor_df_fn,
                                                config=self.config,
                                                debug=debug)
            next_r = next_r.reshape(n_sample, n_look_ahead, cvx_num)
            distractor_rew = distractor_rew.reshape(n_sample, n_look_ahead, cvx_num)
            reward_seqs = next_r[:, -1] + distractor_rew.sum(axis = 1)

        # reward_seqs: n_sample
        # next_r: n_sample, n_look_ahead
        if DEBUG:
            # Output check begin
            print("check output for evaluate_traj")
            print("reward_seqs.shape", reward_seqs.shape)
            assert type(reward_seqs) == torch.Tensor
            print("next_r.shape", next_r.shape)
            assert type(next_r) == torch.Tensor
            print('-----------------')
        assert reward_seqs.shape == (n_sample, cvx_num)
        assert next_r.shape == (n_sample, n_look_ahead, cvx_num)
        return reward_seqs, next_r

    def evaluate_traj_backup(self, obs_seqs, obs_goal, tensor):
        """
        Computes the reward as negative of l2 distance between obs_seqs[:, -1] and goal
        :param obs_seqs:
        :type obs_seqs:
        :param obs_goal:
        :type obs_goal:
        :return:
        :rtype:
        """
        # obs_seqs: [n_sample, n_look_ahead, state_dim]
        # obs_goal: state_dim

        if tensor:
            reward_seqs = -torch.sum((obs_seqs[:, -1] - obs_goal)**2, 1)
        else:
            reward_seqs = -np.sum((obs_seqs[:, -1] - obs_goal)**2, 1)

        # reward_seqs: n_sample
        return reward_seqs

    def optimize_action(self,
                        act_seqs,  # shape: [n_sample, -1, action_dim] / [n_sample, -1, cvx_num, action_dim]
                        reward_seqs  # shape: [n_sample] / [n_sample, cvx_num]
                        ):
        reward_weight = self.config['mpc']['mppi']['reward_weight']
        act_seqs_dim = len(act_seqs.shape)
        assert act_seqs_dim == 4
        n_sample, n_look_ahead, cvx_num, action_dim = act_seqs.shape
        act_seq = np.zeros((n_look_ahead, cvx_num, action_dim))
        for i in range(cvx_num):
            reward_seqs_weights = softmax(reward_weight * reward_seqs[:, i]).reshape(-1, 1, 1)
            act_seq[:, i, :] = (reward_seqs_weights * act_seqs[:, :, i, :]).sum(0)
        return act_seq

    def trajectory_optimization_ptcl_multi_traj(self,
                                                state_cur_np,  # current state, shape: [n_batch, particle_num, 3] numpy array
                                                state_param, # state_param, shape: [n_batch,]
                                                attr_cur_np, # current state, shape: [n_batch, particle_num] numpy array
                                                obs_goal,  # goal, shape: [H, W] numpy array
                                                model_dy,  # the learned dynamics model
                                                act_seq,  # initial action sequence, shape: [-1, traj_num, action_dim] numpy array
                                                act_label_seq,  # initial action sequence, shape: [-1] numpy array
                                                n_sample,  # number of action sequences to sample for each update iter
                                                n_look_ahead,  # number of look ahead steps
                                                n_update_iter,  # number of update iteration
                                                action_lower_lim,
                                                action_upper_lim,
                                                use_gpu=True,
                                                rollout_best_action_sequence=True,
                                                reward_params=None,
                                                funnel_dist=None,
                                                distractor_df_fn=None,
                                                gd_loop=1,
                                                time_lim=float('inf'), # unit: ms
                                                ):
        """

        act_seq has dimensions [n_his + n_look_ahead, action_dim]

        so act_seq[:n_his] matches up with state_cur
        """
        time_lim = time_lim / 1000.0
        reach_time_lim = False
        total_time = 0.0
        assert type(state_cur_np) == np.ndarray
        assert len(state_cur_np.shape) == 3
        assert state_cur_np.shape[0] == state_param.shape[0]
        assert state_cur_np.shape[2] == 3
        
        assert type(obs_goal) == np.ndarray
        assert len(obs_goal.shape) == 2

        assert type(act_seq) == np.ndarray
        assert len(act_seq.shape) == 3
        assert act_seq.shape[0] == act_label_seq.shape[0]
        assert len(act_label_seq.shape) == 1

        assert type(state_param) == np.ndarray
        # assert state_param.shape == (self.n_his,)

        self.particle_num = state_cur_np.shape[1]
        n_batch = state_cur_np.shape[0]

        if use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        state_cur_tensor = torch.tensor(state_cur_np, device=device, dtype=torch.float)
        attr_cur_tensor = torch.tensor(attr_cur_np, device=device, dtype=torch.float)
        obs_goal_tensor = torch.tensor(obs_goal, device=device, dtype=torch.float)
        obs_goal_coor_tensor = torch.flip((obs_goal_tensor < 0.5).nonzero(), dims=(1,)).to(device=device, dtype=torch.float)
        obs_goal_coor_np, _ = fps_np(obs_goal_coor_tensor.detach().cpu().numpy(), min(self.particle_num * 5, obs_goal_coor_tensor.shape[0]), 0)
        # plt.scatter(obs_goal_coor_np[:, 1], obs_goal_coor_np[:, 0])
        # plt.show()
        obs_goal_coor_tensor = torch.tensor(obs_goal_coor_np, device=device, dtype=torch.float)
        state_param_tensor = torch.from_numpy(state_param).to(device=device, dtype=torch.float)

        n_act = act_seq.shape[0]
        traj_num = int(act_seq.shape[1])
        assert n_act == n_look_ahead # the number of actions cannot be less than n_look_ahead
        if DEBUG:
            # Input check begin
            print('-----------------')
            print("check input for model_rollout")
            print("state_cur_np.shape", state_cur_np.shape)
            # viz the current state
            # print('viz the current state')
            # plt.subplot(1, 2, 1)
            # plt.imshow(state_cur_np[0].reshape(4, 64, 64)[3])
            # plt.subplot(1, 2, 2)
            # plt.imshow(state_cur_np[0].reshape(4, 64, 64)[:3].transpose(1, 2, 0))
            # plt.show()
            print("act_seqs_tensor.shape", act_seqs_tensor.shape)
            print('n_look_ahead', n_look_ahead)
            print()
            # Input check end

        rew_mean = np.zeros((1, n_update_iter * gd_loop), dtype=np.float32)
        rew_std = np.zeros((1, n_update_iter * gd_loop), dtype=np.float32)

        # transform goal to pytorch tensor
        # obs_goal_tensor = torch.tensor(obs_goal, device=device, dtype=torch.float, requires_grad=True)
        # obs_goal_tensor = obs_goal

        optim_start = torch.cuda.Event(enable_timing=True)
        optim_end = torch.cuda.Event(enable_timing=True)
        optim_time = 0.0
        rollout_time = 0.0
        oom_error = False

        # redudant model_rollout to avoid overhead
        act_seqs = act_seq.transpose(1, 0, 2)[:, :, np.newaxis, :] # shape: [traj_num, n_act, 1, action_dim]
        act_seqs = np.repeat(act_seqs, n_batch, axis=0) # shape: [traj_num * n_batch, n_act, 1, action_dim]
        act_seqs_tensor = torch.tensor(act_seqs, device=device, dtype=torch.float, requires_grad=True)
        reward_seqs_tensor = torch.ones((traj_num * n_batch, 1), device=device, dtype=torch.float)
        # act_seqs_tensor_mdl_inp = act_seqs_tensor.permute(0, 2, 1, 3).reshape(-1, n_act, self.action_dim)
        # _ = self.ptcl_model_rollout(state_cur_tensor,
        #                             state_param_tensor,
        #                             attr_cur_tensor,
        #                             model_dy, 
        #                             act_seqs_tensor_mdl_inp,
        #                             enable_grad = True)
        start = time.time()

        optimizer = optim.Adam([act_seqs_tensor], lr=self.config['mpc']['gd']['lr'], betas=(0.9, 0.999))

        max_reward = -float('inf') * torch.ones(n_batch, device=device, dtype=torch.float)
        max_reward_traj_idx = torch.zeros(n_batch, device=device, dtype=torch.long)
        best_actions_of_samples = torch.zeros((n_batch, n_act, self.action_dim), device=device, dtype=torch.float)
        iter_bound_by_time = int(time_lim * 1000.0/ particle_num_to_iter_time(self.particle_num))
        print('run mpc for {} iterations'.format(min(n_update_iter, iter_bound_by_time)))

        for i in range(min(n_update_iter, iter_bound_by_time)):
            # rollout using the sampled action sequences and the learned model
            # [n_samples, n_act, 1, action_dim]
            act_seqs_tensor_mdl_inp = act_seqs_tensor.permute(0, 2, 1, 3).reshape(-1, n_act, self.action_dim)
            try:
                out = self.ptcl_model_rollout(
                        state_cur_tensor,
                        state_param_tensor,
                        attr_cur_tensor,
                        model_dy, 
                        act_seqs_tensor_mdl_inp,
                        enable_grad = True)
            except:
                print('OOM error')
                break
            out['model_rollout']['state_pred'] = out['model_rollout']['state_pred'].reshape(n_sample * n_batch, 1, n_act, self.particle_num, 3).permute(0, 2, 1, 3, 4)
            rollout_time += out['rollout_time']

            obs_seqs_tensor = out['model_rollout']['state_pred'] # (n_sample, 1, n_act, self.particle_num, 3)

            reward_seqs_tensor, _ = self.ptcl_evaluate_traj(
                obs_seqs_tensor,
                obs_goal_tensor,
                obs_goal_coor_tensor,
                distractor_df_fn=distractor_df_fn,
                act_seqs_tensor=act_seqs_tensor,
                ) #  [n_sample * n_batch, 1]

            # # print top k trajectories
            # reward_seqs_tensor_np = reward_seqs_tensor.detach().cpu().numpy()
            # # choose top k amoong reward_seqs_tensor_np
            # top_k = 5
            # top_k_idx = np.argsort(reward_seqs_tensor_np, axis=0)[-top_k:]
            # print('top %d trajectories' % top_k)
            # print('index', top_k_idx)
            # print('reward', reward_seqs_tensor_np[top_k_idx, 0])
            # print('action', act_seqs_tensor[top_k_idx, 0, 0].detach().cpu().numpy())
            
            # aggregation
            reward_seqs_tensor = reward_seqs_tensor.reshape(n_sample, n_batch)
            curr_max_reward, idx_best_act = torch.max(reward_seqs_tensor, dim=0)
            for j in range(n_batch):
                if curr_max_reward[j] > max_reward[j]:
                    max_reward[j] = curr_max_reward[j]
                    max_reward_traj_idx[j] = idx_best_act[j]
                    best_actions_of_samples[j] = act_seqs_tensor[idx_best_act[j] * n_batch + j, :, 0]
            # act_seqs = act_seqs_tensor.data.cpu().numpy()
            # idx_best_act = torch.argmax(reward_seqs_tensor).item()
            # act_seq = act_seqs[idx_best_act, :, np.arange(1)].transpose(1, 0, 2) # [n_act, 1, action_dim]

            if DEBUG:
                print('update_iter %d/%d, max: %.4f, mean: %.4f, std: %.4f' % (
                    i, n_update_iter, torch.max(reward_seqs_tensor),
                    torch.mean(reward_seqs_tensor), torch.std(reward_seqs_tensor)))
            for cvx_i in range(1):
                rew_mean[cvx_i, i] = torch.mean(reward_seqs_tensor[:, cvx_i]).item()
                rew_std[cvx_i, i] = torch.std(reward_seqs_tensor[:, cvx_i]).item()

            # optimize the action sequence according to the rewards
            try:
                optim_start.record()
                loss = torch.sum(-reward_seqs_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optim_end.record()
            except:
                print('OOM error')
                break

            torch.cuda.synchronize()
            optim_time += optim_start.elapsed_time(optim_end)

            # clip to the lower and upper limits
            for cvx_i in range(1):
                x_diff = self.env.cvx_region[cvx_i, 1] - self.env.cvx_region[cvx_i, 0]
                y_diff = self.env.cvx_region[cvx_i, 3] - self.env.cvx_region[cvx_i, 2]
                cvx_lower_lim = np.array([self.env.cvx_region[cvx_i, 0], self.env.cvx_region[cvx_i, 2], self.env.cvx_region[cvx_i, 0] + x_diff * 0.15, self.env.cvx_region[cvx_i, 2] + y_diff * 0.15])
                cvx_upper_lim = np.array([self.env.cvx_region[cvx_i, 1], self.env.cvx_region[cvx_i, 3], self.env.cvx_region[cvx_i, 1] - x_diff * 0.15, self.env.cvx_region[cvx_i, 3] - y_diff * 0.15])
                act_seqs_tensor.data[:, :, cvx_i, 0].clamp_(min=cvx_lower_lim[0], max=cvx_upper_lim[0])
                act_seqs_tensor.data[:, :, cvx_i, 1].clamp_(min=cvx_lower_lim[1], max=cvx_upper_lim[1])
                act_seqs_tensor.data[:, :, cvx_i, 2].clamp_(min=cvx_lower_lim[2], max=cvx_upper_lim[2])
                act_seqs_tensor.data[:, :, cvx_i, 3].clamp_(min=cvx_lower_lim[3], max=cvx_upper_lim[3])
            # print('action_sequence_optimized', act_seq)
            # input()
            # if (time.time() - start) > time_lim:
            #     print('reach time limit')
            #     break
        # aggregation
        reward_seqs = reward_seqs_tensor.data.cpu().numpy()
        act_seqs = act_seqs_tensor.data.cpu().numpy()
        max_reward_traj_count = torch.bincount(max_reward_traj_idx)
        idx_best_act = torch.argmax(max_reward_traj_count).item()
        idx_best_sample = -1
        reward_from_best_sample = - float('inf')
        for j in range(n_batch):
            if idx_best_act == max_reward_traj_idx[j] and max_reward[j] > reward_from_best_sample:
                idx_best_sample = j
                reward_from_best_sample = max_reward[j]
        act_seq = best_actions_of_samples.detach().cpu().numpy()[idx_best_sample][:, None, :] # [n_act, 1, action_dim]
        # idx_best_state = np.argmax(reward_seqs[idx_best_act])
        
        # act_seq = act_seqs[idx_best_act * n_batch + idx_best_state, :, np.arange(1)].transpose(1, 0, 2) # [n_act, 1, action_dim]
        # idx_best_act = np.argmax(reward_seqs, axis=0) # [1]
        # act_seq = act_seqs[idx_best_act, :, np.arange(1)].transpose(1, 0, 2) # [n_act, 1, action_dim]
        # act_seq = self.optimize_action(act_seqs, reward_seqs) # [n_act, 1, action_dim]
        if DEBUG:
            plt.subplot(2, 2, 1)
            plt.plot(rew_mean[0])
            plt.fill_between(range(n_update_iter), rew_mean[0] - rew_std[0], rew_mean[0] + rew_std[0], alpha=0.5)
            plt.xlabel('update iteration')
            plt.ylabel('reward')
            plt.title('reward for convex region 1 [left]')
            plt.subplot(2, 2, 2)
            plt.plot(rew_mean[1])
            plt.xlabel('update iteration')
            plt.ylabel('reward')
            plt.title('reward for convex region 2 [middle]')
            plt.fill_between(range(n_update_iter), rew_mean[1] - rew_std[1], rew_mean[1] + rew_std[1], alpha=0.5)
            plt.subplot(2, 2, 3)
            plt.plot(rew_mean[2])
            plt.xlabel('update iteration')
            plt.ylabel('reward')
            plt.title('reward for convex region 3 [right]')
            plt.fill_between(range(n_update_iter), rew_mean[2] - rew_std[2], rew_mean[2] + rew_std[2], alpha=0.5)
            plt.subplot(2, 2, 4)
            plt.plot(rew_mean[3])
            plt.xlabel('update iteration')
            plt.ylabel('reward')
            plt.title('reward for convex region 4 [top]')
            plt.fill_between(range(n_update_iter), rew_mean[3] - rew_std[3], rew_mean[3] + rew_std[3], alpha=0.5)
            plt.show()
            plt.close()

        # observation sequence for the best action sequence
        # that was found
        obs_seq_best = None
        reward_best = None
        obs_seq_distractor_best = None
        if rollout_best_action_sequence:
            act_seq = act_seq.transpose(1, 0, 2)
            assert act_seq.shape[0] == 1
            assert act_seq.shape[1] == n_act
            assert act_seq.shape[2] == self.action_dim
            act_seq_tensor = torch.from_numpy(act_seq).float().cuda()
            out = self.ptcl_model_rollout(
                state_cur_tensor[0:1],
                state_param_tensor[0:1],
                attr_cur_tensor[0:1],
                model_dy, 
                act_seq_tensor,
                enable_grad = True)
            # [1, n_look_ahead, particle_num, 3]

            obs_seq_best = out['model_rollout']['state_pred'].permute(1, 0, 2, 3).unsqueeze(0)
            # [1, n_look_ahead + 1, 1, particle_num, 3]

            # reward_seq_best: [1]
            # next_seq_r: [1, n_look_ahead]
            reward_seq_best, next_seq_r = self.ptcl_evaluate_traj(
                obs_seq_best,
                obs_goal_tensor,
                obs_goal_coor_tensor,
                distractor_df_fn=distractor_df_fn,
                act_seqs_tensor=act_seq_tensor[None, ...],
            )
            reward_best_idx = next_seq_r[:, 0].argmax()
            next_r = next_seq_r[reward_best_idx]
            reward_best = reward_seq_best[reward_best_idx]
            obs_seq_best = out['model_rollout']['state_pred'][reward_best_idx].detach().cpu().numpy()
        action_seq_future = act_seq[reward_best_idx]

        # Check output starts
        end = time.time()
        total_time = end - start

        return {'action_sequence': action_seq_future,   # [n_roll, action_dim]
                'action_full': act_seqs[:, 0, 0, :], # [traj_num, n_act, action_dim]
                'reward_full': reward_seqs[:, 0], # [traj_num]
                'observation_sequence': obs_seq_best,   # [n_roll, particle_num, 3]
                'observation_distractor_sequence': obs_seq_distractor_best,   # [n_roll, obs_dim]
                'reward': reward_best.detach().cpu().numpy(),
                'next_r': next_r.detach().cpu().numpy(),
                'rew_mean': rew_mean,
                'rew_std': rew_std,
                'times': {'total_time': total_time,
                          'rollout_time': rollout_time,
                          'optim_time': optim_time,},
                'iter_num': i,
                }
