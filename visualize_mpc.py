# Using new version

import os
import cv2
import pickle
import numpy as np
from env.flex_env import FlexEnv
import multiprocessing as mp
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
from scipy.special import softmax

# utils
from utils import load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed, pcd2pix, gen_goal_shape, gen_subgoal, gt_rewards, gt_rewards_norm_by_sum, lighten_img, rmbg
from model.gnn_dyn import PropNetDiffDenModel

def main():
    config = load_yaml("config/mpc/config.yaml")

    model_folder = config['mpc']['model_folder']
    model_iter = config['mpc']['iter_num']
    n_mpc = config['mpc']['n_mpc']
    n_look_ahead = config['mpc']['n_look_ahead']
    n_sample = config['mpc']['n_sample']
    n_update_iter = config['mpc']['n_update_iter']
    gd_loop = config['mpc']['gd_loop']
    mpc_type = config['mpc']['mpc_type']

    task_type = config['mpc']['task']['type']

    model_root = 'data/gnn_dyn/'
    model_folder = os.path.join(model_root, model_folder)
    GNN_single_model = PropNetDiffDenModel(config, True)
    if model_iter == -1:
        GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_best.pth'), strict=False)
    else:
        GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_epoch_0_iter_{model_iter}.pth'), strict=False)
    GNN_single_model = GNN_single_model.cuda()

    env = FlexEnv(config)
    screenWidth = screenHeight = 720

    if task_type == 'target_control':
        goal_row = config['mpc']['task']['goal_row']
        goal_col = config['mpc']['task']['goal_col']
        goal_r = config['mpc']['task']['goal_r']
        subgoal, mask = gen_subgoal(goal_row,
                                    goal_col,
                                    goal_r,
                                    h=screenHeight,
                                    w=screenWidth)
        goal_img = (mask[..., None]*255).repeat(3, axis=-1).astype(np.uint8)
    elif task_type == 'target_shape':
        goal_char = config['mpc']['task']['target_char']
        subgoal, goal_img = gen_goal_shape(goal_char,
                                            h=screenHeight,
                                            w=screenWidth)
    else:
        raise NotImplementedError
    
    env.reset()

    funnel_dist = np.zeros_like(subgoal)

    action_seq_mpc_init = np.load('init_action/init_action_'+ str(n_sample) +'.npy')[np.newaxis, ...] # [1, 50, 4]
    action_label_seq_mpc_init = np.zeros(1)
    subg_output = env.step_subgoal_ptcl(subgoal,
                                        GNN_single_model,
                                        None,
                                        n_mpc=n_mpc,
                                        n_look_ahead=n_look_ahead,
                                        n_sample=n_sample,
                                        n_update_iter=n_update_iter,
                                        mpc_type=mpc_type,
                                        gd_loop=gd_loop,
                                        particle_num=-1,
                                        funnel_dist=funnel_dist,
                                        action_seq_mpc_init=action_seq_mpc_init,
                                        action_label_seq_mpc_init=action_label_seq_mpc_init,
                                        time_lim=config['mpc']['time_lim'],
                                        auto_particle_r=True,)

if __name__ == "__main__":
    main()
