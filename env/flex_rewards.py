import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import sys
# from pytorch3d.loss import chamfer_distance
sys.path.append('.')
from utils import fps_np, pcd2pix

def depth_to_pcd(depth, original_size, cam_params, cam_extrinsic):
    # depth: (obs_w, obs_h), in original depth
    # original_size: (w, h)
    # cam_params: (fx, fy, cx, cy)
    fx, fy, cx, cy = cam_params
    w, h = original_size
    obs_w, obs_h = depth.shape
    fx = fx * obs_w / w
    fy = fy * obs_h / h
    cx = cx * obs_w / w
    cy = cy * obs_h / h
    # resized_depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA)
    resized_depth = depth
    x, y = np.meshgrid(np.arange(obs_w), np.arange(obs_h))
    x = ((x - cx) * resized_depth / fx).reshape(-1)
    y = ((y - cy) * resized_depth / fy).reshape(-1)
    z = resized_depth.reshape(-1)
    pcd = np.stack([x, y, z], axis=1)
    pcd = np.matmul(cam_extrinsic, np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T).T[:, :3] # H*W, 3
    pcd = pcd.reshape(obs_h, obs_w, 3)
    return pcd

def depth_to_pcd_torch(depth, original_size, cam_params, cam_extrinsic):
    # depth: (obs_w, obs_h), in original depth
    # original_size: (w, h)
    # cam_params: (fx, fy, cx, cy)
    fx, fy, cx, cy = cam_params
    w, h = original_size
    obs_w, obs_h = depth.shape
    fx = fx * obs_w / w
    fy = fy * obs_h / h
    cx = cx * obs_w / w
    cy = cy * obs_h / h
    # resized_depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA)
    resized_depth = depth
    x, y = np.meshgrid(np.arange(obs_w), np.arange(obs_h))
    x = ((x - cx) * resized_depth / fx).reshape(-1)
    y = ((y - cy) * resized_depth / fy).reshape(-1)
    z = resized_depth.reshape(-1)
    pcd = np.stack([x, y, z], axis=1)
    pcd = np.matmul(cam_extrinsic, np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T).T[:, :3] # H*W, 3
    pcd = pcd.reshape(obs_h, obs_w, 3)
    return pcd

def highest_reward(state, reward_params):
    # state: (N, C, H, W) torch tensor / numpy array
    # output: (N, ) torch tensor / numpy array
    cam_extrinsic, cam_params, global_scale = reward_params
    N, C, H, W = state.shape
    is_torch = (type(state) == torch.Tensor)
    if (is_torch):
        state_arr = state.detach().cpu().numpy()
    else:
        state_arr = state
    
    # convert to point cloud
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    opencv_T_world = np.matmul(np.linalg.inv(cam_extrinsic), opencv_T_opengl)
    original_size = (720, 720)
    if (is_torch):
        rewards = torch.zeros(N)
    else:
        rewards = np.zeros(N)
    for i in range(N):
        state_3d = depth_to_pcd(state_arr[i, -1] * global_scale, original_size, cam_params, opencv_T_world)
        mask = np.ones((H, W))
        # mask[state_3d[..., 1] < 0.05] = 0
        # mask[state_3d[..., 0] > 1.85] = 0
        # mask[state_3d[..., 0] < -1.85] = 0
        # mask[state_3d[..., 2] > 1.85] = 0
        # mask[state_3d[..., 2] < -1.85] = 0
        if (is_torch):
            mask = torch.from_numpy(mask).float().to(state.device)
            rewards[i] = torch.amax(-state[i, -1, :, :][mask == 1])
        else:
            rewards[i] = np.amax(-state[i, -1, :, :][mask == 1])
    return rewards

def get_hsv(im):
    eps = 1e-7
    img = im * 0.5 + 0.5
    hsv = torch.zeros_like(im).to(im.device)
    hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]
    hsv[:, 0, :, :] = hue
    hsv[:, 1, :, :] = saturation
    hsv[:, 2, :, :] = value
    return hsv


def config_reward(state, goal, goal_mask, img_format, debug=False, save_folder=None):
    # input:
    #  - state: (N, C, H, W) torch tensor (depth channel is scaled by global_scaling and 1000.)
    #  - goal: (H, W) torch tensor of goal image
    # output: (N, ) torch tensor / numpy array
    N, C, H, W = state.shape
    assert type(state) == torch.Tensor
    assert type(goal) == torch.Tensor
    assert type(goal_mask) == torch.Tensor
    assert goal.shape == (H, W)
    assert goal_mask.shape == (H, W)
    assert C == 1

    if img_format == 'binary':
        mask = state[:, 0, :, :]
        binary_mask = (mask > 0.5).float()
        binary_mask_np = binary_mask.detach().cpu().numpy().astype(np.uint8)
        state_dists = []
        for i in range(N):
            state_dists.append(np.minimum(cv2.distanceTransform((1 - binary_mask_np[i]), cv2.DIST_L2, 5), 1e4))
        state_dist = np.stack(state_dists, axis=0)
        state_dist = torch.from_numpy(state_dist).float().to(state.device) # (N, H, W)
    else:
        raise NotImplementedError

    # plt.subplot(2, 2, 1)
    # plt.imshow(mask[0].detach().cpu().numpy())
    # plt.subplot(2, 2, 2)
    # plt.imshow(goal.detach().cpu().numpy())
    # plt.subplot(2, 2, 3)
    # plt.imshow(state_dist[0].detach().cpu().numpy())
    # plt.subplot(2, 2, 4)
    # plt.imshow(goal_mask.detach().cpu().numpy())
    # plt.show()
    rewards = (goal[None, ...].repeat(N, 1, 1) * mask).sum(dim=(1, 2)) # + (state_dist * goal_mask[None, ...].repeat(N, 1, 1)).sum(dim=(1, 2))
    if debug:
        print('config reward', rewards)
    return -rewards

def config_reward_ptcl(state, goal, cam_params, goal_coor, normalize=True, offset=(0., 0.)):
    # input:
    #  - state: (B, N, 3) torch tensor of particles
    #  - goal: (H, W) torch tensor of goal image
    #  - cam_params: (4, ) torch tensor of camera parameters
    #  - goal_coor: (M, 2) torch tensor of goal coordinates
    #  - offset: (2, ) tuple of offset for pixel piositions after transforming, (col, row)
    # output: (B, ) torch tensor
    B, N, _ = state.shape
    H, W = goal.shape
    assert state.shape[2] == 3
    assert type(state) == torch.Tensor
    assert type(goal) == torch.Tensor
    # assert goal_coor.shape[0] == N
    fx, fy, cx, cy = cam_params
    # goal_coor = torch.flip((goal < 0.5).nonzero(), dims=(1,)) # (M, 2) where 2 is (col, row)
    goal_np = goal.detach().cpu().numpy()
    goal_seg = (goal_np < 0.5)
    neg_goal_dist = cv2.distanceTransform(goal_seg.astype(np.uint8), cv2.DIST_L2, 5)
    goal_np = goal_np - neg_goal_dist
    goal_np = goal_np - goal_np.min()
    goal = torch.from_numpy(goal_np).to(device = state.device, dtype = state.dtype)
    goal = torch.tile(goal[None, None, ...], (B, 1, 1, 1)) # (B, 1, H, W)

    # downsample goal_coor to the same scale as state
    # goal_coor_idx = farthest_point_sampler(goal_coor[None, ...].to(device = 'cpu', dtype = torch.float32), N)[0]
    # goal_coor_idx = goal_coor_idx.to(device = state.device)
    # goal_coor = goal_coor[goal_coor_idx] # (N, 2)
    # goal_coor_np = fps_np(goal_coor.detach().cpu().numpy(), N)
    # goal_coor = torch.from_numpy(goal_coor_np).to(device = state.device, dtype = state.dtype)
    goal_coor = torch.tile(goal_coor[None, ...], (B, 1, 1)) # (B, M, 2) # (col, row)

    # convert to pixel space
    pix = torch.zeros((B, N, 2)).to(state.device, state.dtype) # (B, N, 2), where 2 is (col, row)
    pix[:, :, 0] = state[:, :, 0] * fx / state[:, :, 2] + cx # TODO: maybe need to remove grad thru state[:, :, 2]
    pix[:, :, 1] = state[:, :, 1] * fy / state[:, :, 2] + cy
    pix[:, :, 0] += offset[0]
    pix[:, :, 1] += offset[1]
    pix.unsqueeze_(1) # (B, 1, N, 2)

    # read from goal
    norm_pix = pix / H * 2 - 1 # (B, 1, N, 2)
    rewards = F.grid_sample(goal, norm_pix, padding_mode="border", align_corners=False) # (B, 1, 1, N)
    rewards = rewards.squeeze(1).squeeze(1).sum(dim = 1) # (B,)
    
    # read distance from goal to pix
    # plt.subplot(1, 2, 1)
    # plt.scatter(goal_coor.detach().cpu().numpy()[0, :, 0], goal_coor.detach().cpu().numpy()[0, :, 1])
    # plt.subplot(1, 2, 2)
    # plt.scatter(pix.detach().cpu().numpy()[0, 0, :, 0], pix.detach().cpu().numpy()[0, 0, :, 1])
    # plt.show()
    dist_to_pix = torch.norm(goal_coor[:, :, None, :] - pix, dim = 3) # (B, M, N)
    min_dist_to_pix, _ = dist_to_pix.min(dim = 2) # (B, M)
    rewards += min_dist_to_pix.sum(dim = 1) # (B,)

    if normalize:
        rewards /= N # normalize across different particle size

    return -rewards

def distractor_reward(act_seqs_tensor, distractor_dist, config, debug=False):
    if type(distractor_dist) == np.ndarray:
        distractor_dist_tensor = torch.from_numpy(distractor_dist).float().to('cuda')
    elif type(distractor_dist) == torch.Tensor:
        distractor_dist_tensor = distractor_dist
    else:
        raise NotImplementedError

    N = act_seqs_tensor.shape[0] * act_seqs_tensor.shape[1]
    act_seqs_tensor_img_spc = act_seqs_tensor.clone()
    act_seqs_tensor_img_spc[..., 0] = -act_seqs_tensor[..., 1] / config['dataset']['wkspc_w'] * config['dataset']['state_h'] / 3. + config['dataset']['state_h'] / 2.
    act_seqs_tensor_img_spc[..., 1] = act_seqs_tensor[..., 0] / config['dataset']['wkspc_w'] * config['dataset']['state_h'] / 3. + config['dataset']['state_h'] / 2.
    act_seqs_tensor_img_spc[..., 2] = -act_seqs_tensor[..., 3] / config['dataset']['wkspc_w'] * config['dataset']['state_h'] / 3. + config['dataset']['state_h'] / 2.
    act_seqs_tensor_img_spc[..., 3] = act_seqs_tensor[..., 2] / config['dataset']['wkspc_w'] * config['dataset']['state_h'] / 3. + config['dataset']['state_h'] / 2.
    _, _, cvx_num, act_dim = act_seqs_tensor_img_spc.shape
    max_dist = torch.zeros(N).float().cuda()
    for i in range(10 + 1):
        w = i / 10.
        act_seqs_tensor_img_spc = act_seqs_tensor_img_spc.reshape(N, cvx_num, act_dim)
        pixel_x = (act_seqs_tensor_img_spc[:, 0, 0] * w + act_seqs_tensor_img_spc[:, 0, 2] * (1 - w)).long()
        pixel_y = (act_seqs_tensor_img_spc[:, 0, 1] * w + act_seqs_tensor_img_spc[:, 0, 3] * (1 - w)).long()
        max_dist = torch.max(distractor_dist_tensor[pixel_x, pixel_y], max_dist)

    if debug:
        print('max_dist: ', max_dist)

    return -max_dist


def distractor_reward_diff(act_seqs_tensor, distractor_dist_fn, config, debug=False, width=64):
    N = act_seqs_tensor.shape[0] * act_seqs_tensor.shape[1]
    act_seqs_tensor_img_spc = act_seqs_tensor.clone()
    act_seqs_tensor_img_spc[..., 0] = -act_seqs_tensor[..., 1] / config['dataset']['wkspc_w'] * width / 3. +  width / 2.
    act_seqs_tensor_img_spc[..., 1] = act_seqs_tensor[..., 0] / config['dataset']['wkspc_w'] * width / 3. +  width / 2.
    act_seqs_tensor_img_spc[..., 2] = -act_seqs_tensor[..., 3] / config['dataset']['wkspc_w'] * width / 3. +  width / 2.
    act_seqs_tensor_img_spc[..., 3] = act_seqs_tensor[..., 2] / config['dataset']['wkspc_w'] * width / 3. + width / 2.
    _, _, cvx_num, act_dim = act_seqs_tensor_img_spc.shape
    max_dist = torch.zeros(N).float().cuda()
    for i in range(10 + 1):
        w = i / 10.
        act_seqs_tensor_img_spc = act_seqs_tensor_img_spc.reshape(N, cvx_num, act_dim)
        pixel_x = (act_seqs_tensor_img_spc[:, 0, 0] * w + act_seqs_tensor_img_spc[:, 0, 2] * (1 - w))
        pixel_y = (act_seqs_tensor_img_spc[:, 0, 1] * w + act_seqs_tensor_img_spc[:, 0, 3] * (1 - w))
        max_dist = torch.max(distractor_dist_fn(pixel_x, pixel_y), max_dist)

    if debug:
        print('max_dist: ', max_dist)

    return -max_dist
