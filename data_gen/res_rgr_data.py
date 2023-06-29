import os

import torch
import cv2

import numpy as np
np.int = np.int64 # work around for issue https://github.com/scikit-optimize/scikit-optimize/issues/1171
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize, gbrt_minimize
from skopt import Optimizer
from skopt.learning.gaussian_process.kernels import WhiteKernel, RBF, Matern
from skopt.learning import GaussianProcessRegressor, GradientBoostingQuantileRegressor
from skopt.space import Integer, Categorical
from utils import load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed, pcd2pix, rmbg, gt_rewards, gen_subgoal, gen_goal_shape
from env.flex_env import FlexEnv
from model.gnn_dyn import PropNetDiffDenModel
from skopt.plots import plot_convergence
from skopt.utils import expected_minimum
from skopt import dump, load
from scipy.special import softmax
from sklearn.ensemble import GradientBoostingRegressor

import argparse

def save_obs(obs, save_dir, idx, vid=None):
    assert obs.shape[2] == 5 # make sure the obs is [RGBA, depth]
    assert obs[..., :3].max() <= 255.0 # make sure the color is in [0, 255]
    assert obs[..., :3].max() >= 1.0 # make sure the color is in [0, 255]
    assert obs[..., :3].min() >= 0.0 # make sure the color is in [0, 255]
    cv2.imwrite(os.path.join(save_dir, "color_%d.png" % idx), obs[..., :3][..., ::-1].astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, "depth_%d.png" % idx), (obs[..., -1]*1000).astype(np.uint16))
    if vid is not None:
        vid.write(obs[..., :3][..., ::-1].astype(np.uint8))

def random_angle_steps(steps: int, irregularity: float) -> np.array:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        np.array: the random angles.
    """
    # generate n angle steps
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    angles = np.random.uniform(lower, upper, steps)
    angles = angles / np.sum(angles) * 2 * np.pi
    return angles
    
def gen_rand_polygon(h, w):
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.
    """
    center = np.array([h / 2., w / 2.])
    avg_radius = np.random.uniform(50., 150.)
    num_vertices = np.random.randint(8, 13)
    irregularity = np.random.uniform(0, 1.5 * np.pi / num_vertices)
    spikiness = np.random.uniform(0, 0.8 * avg_radius)

    angle_steps = random_angle_steps(num_vertices, irregularity)
    angles = np.zeros_like(angle_steps)
    for i in range(1, len(angle_steps)):
        angles[i] = angles[i - 1] + angle_steps[i]

    # now generate the points
    points = np.zeros((num_vertices, 2))
    angles = np.random.uniform(0, 2 * np.pi) + angles # rotate it
    radius = np.clip(np.random.normal(avg_radius, spikiness, num_vertices), 0.1 * avg_radius, 2 * avg_radius)
    points[:, 0] = center[0] + radius * np.cos(angles)
    points[:, 1] = center[1] + radius * np.sin(angles)
    points += np.random.uniform(-100, 100, 2)
    points = points.astype(np.int32)
    
    goal_img = np.zeros((int(h), int(w), 3)).astype(np.uint8)
    cv2.fillPoly(goal_img, pts = [points], color =(255,255,255))

    goal_dist = cv2.distanceTransform(((255-goal_img[..., 0])/255).astype(np.uint8), cv2.DIST_L2, 5)

    # plt.subplot(1,2,1)
    # plt.imshow(goal_img)
    # plt.subplot(1,2,2)
    # plt.imshow(goal_dist)
    # plt.show()
    # plt.close()

    return goal_dist, goal_img

class GPParamOpt():
    
    def __init__(self):
        self.config = load_yaml("config/data_gen/res_rgr.yaml")

        self.global_scale = self.config['dataset']['global_scale']
        model_folder = self.config['mpc']['model_folder']
        self.n_mpc = self.config['mpc']['n_mpc']
        self.n_mpc_per_model = self.config['mpc']['n_mpc_per_model']
        self.num_steps = self.n_mpc // self.n_mpc_per_model
        self.n_look_ahead = self.config['mpc']['n_look_ahead']
        self.n_sample = self.config['mpc']['n_sample']
        self.n_update_iter = self.config['mpc']['n_update_iter']
        self.gd_loop = self.config['mpc']['gd_loop']
        self.mpc_type = self.config['mpc']['mpc_type']
        self.noise_level = 1e5

        self.env = FlexEnv(self.config)
        self.env.reset()

        # load model
        model_root = 'data/gnn_dyn_model/'
        model_folder = os.path.join(model_root, model_folder)
        self.GNN_model = PropNetDiffDenModel(self.config, True)
        if self.config['mpc']['iter_num'] > -1:
            self.GNN_model.load_state_dict(torch.load('%s/net_epoch_0_iter_%d.pth' % (model_folder, self.config['mpc']['iter_num'])), strict=False)
        else:
            self.GNN_model.load_state_dict(torch.load('%s/net_best.pth' % (model_folder)), strict=False)
        self.GNN_model = self.GNN_model.cuda()

        self.screenWidth = self.screenHeight = 720

    def param_eval_fn(self, particle_num):
        print('------------------------')
        print('evaluation:', self.eval_idx)
        # print('particle density:', particle_den[0])
        # GNN_model_res = 1 / np.sqrt(particle_den[0])
        # GNN_model_density = particle_den[0]
        # particle_num = int(particle_num[0] * 140. + 10.)
        particle_num = int(particle_num[0])
        test_repeat = 5
        res = np.zeros(test_repeat)

        # NOTE: one step is composed of multiple MPC steps
        for i in range(test_repeat):
            self.env.set_positions(self.last_pos)
            subg_output = self.env.step_subgoal_ptcl(self.subgoal,
                                                    self.GNN_model,
                                                    None,
                                                    n_mpc=self.n_mpc_per_model,
                                                    n_look_ahead=self.n_look_ahead,
                                                    n_sample=self.n_sample,
                                                    n_update_iter=self.n_update_iter,
                                                    mpc_type=self.mpc_type,
                                                    gd_loop=self.gd_loop,
                                                    particle_num=particle_num,
                                                    funnel_dist=self.funnel_dist,
                                                    action_seq_mpc_init=self.action_seq_mpc_init,
                                                    action_label_seq_mpc_init=self.action_label_seq_mpc_init,
                                                    time_lim=self.config['mpc']['time_lim'],)
            self.pos.append(self.env.get_positions())
            rew_mean = subg_output['rew_means']
            rew_std = subg_output['rew_stds']
            states_pred = subg_output['states_pred']
            states = subg_output['states']
            rewards = subg_output['rewards']
            raw_obs = subg_output['raw_obs']
            if self.record_mpc_data:
                save_dir_i = os.path.join(self.save_dir_curr_step, str(self.eval_idx) + '_' + str(particle_num))
                os.system('mkdir -p ' + save_dir_i)
                save_dir_i = os.path.join(save_dir_i, str(i))
                os.system('mkdir -p ' + save_dir_i)
                
                # save video and intermediate results
                vid = cv2.VideoWriter(os.path.join(save_dir_i, "mpc.avi"), cv2.VideoWriter_fourcc(*'XVID'), 1, (self.screenWidth, self.screenHeight))

                for mpc_i in range(self.n_mpc_per_model):
                    plt.plot(rew_mean[mpc_i, 0])
                    plt.fill_between(np.arange(self.n_update_iter * self.gd_loop),
                                    rew_mean[mpc_i, 0] - rew_std[mpc_i, 0],
                                    rew_mean[mpc_i, 0] + rew_std[mpc_i, 0],
                                    alpha=0.5)
                    plt.savefig(os.path.join(save_dir_i, 'rew_' + str(mpc_i) + '.png'))
                    plt.close()
                    
                    plt.subplot(3, 2, 1)
                    plt.imshow(self.subgoal + self.funnel_dist)
                    plt.subplot(3, 2, 2)
                    plt.imshow(subg_output['raw_obs'][mpc_i + 1][..., :3].astype(np.uint8))
                    gt_img = np.ones_like(self.goal_img) * 255
                    states_pix = pcd2pix(states[mpc_i + 1], self.env.get_cam_params())
                    for pix_i in range(states_pix.shape[0]):
                        cv2.circle(gt_img, (states_pix[pix_i, 1], states_pix[pix_i, 0]), 5, (255, 0, 0), -1)
                    plt.subplot(3, 2, 3)
                    plt.imshow(gt_img.astype(np.uint8))
                    pred_img = np.ones_like(self.goal_img) * 255
                    states_pred_pix = pcd2pix(states_pred[mpc_i], self.env.get_cam_params())
                    for pix_i in range(states_pred_pix.shape[0]):
                        cv2.circle(pred_img, (states_pred_pix[pix_i, 1], states_pred_pix[pix_i, 0]), 5, (255, 0, 0), -1)
                    plt.subplot(3, 2, 4)
                    plt.imshow(pred_img.astype(np.uint8))
                    plt.subplot(3, 2, (5, 6))
                    plt.plot(rewards[:(mpc_i+1)])
                    plt.ylim([rewards.min() - 1000., rewards.max() + 1000.])
                    plt.xlim([-1, self.n_mpc_per_model+1])
                    plt.savefig(os.path.join(save_dir_i, 'log_' + str(mpc_i) + '.png'))
                    plt.close()

                    save_obs(raw_obs[mpc_i], save_dir_i, mpc_i, vid)
                save_obs(raw_obs[-1], save_dir_i, rew_mean.shape[0], vid)
                for _ in range(3):
                    blended_img = cv2.addWeighted(raw_obs[-1][..., :3][..., ::-1].astype(np.uint8), 0.5, self.goal_img, 0.5, 0)
                    vid.write(blended_img)
                vid.release()
            last_rew = gt_rewards((raw_obs[-1][..., -1] / self.global_scale) < 0.599/0.8, self.subgoal)
            first_rew = gt_rewards((raw_obs[0][..., -1] / self.global_scale) < 0.599/0.8, self.subgoal)
            res[i] = (last_rew - first_rew)
        eval_res = res.mean()
        # eval_res = np.percentile(res, 50)
        print('reg:', 0.001 * first_rew * particle_num)
        eval_res += 0.001 * first_rew * particle_num
        self.eval_idx += 1
        print('res: ', eval_res)
        print('------------------------')
        print()
        return eval_res

    def plot_optimizer(self, res, n_iter, max_iters=5):
        if n_iter == 0:
            show_legend = True
        else:
            show_legend = False
        ax = plt.subplot(max_iters, 2, 2 * n_iter + 1)
        # Plot GP(x) + contours
        ax = plot_gaussian_process(res, ax=ax,
                                noise_level=self.noise_level,
                                show_legend=show_legend, show_title=True,
                                show_next_point=False, show_acq_func=False)
        min_x = res.x
        min_y = res.fun
        approx_x, approx_fn = expected_minimum(res)
        plt.plot(approx_x, approx_fn, 'b.', markersize=10, label=u'Minimum')
        ax.set_ylabel("")
        ax.set_xlabel("")
        if n_iter < max_iters - 1:
            ax.get_xaxis().set_ticklabels([])
        # Plot EI(x)
        ax = plt.subplot(max_iters, 2, 2 * n_iter + 2)
        ax = plot_gaussian_process(res, ax=ax,
                                noise_level=self.noise_level,
                                show_legend=show_legend, show_title=False,
                                show_next_point=True, show_acq_func=True,
                                show_observations=False,
                                show_mu=False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        if n_iter < max_iters - 1:
            ax.get_xaxis().set_ticklabels([])
    
    def plot_final_curve(self, res, ylim=None):
        font = {'family' : 'Times New Roman',
                'weight' : 'normal',
                'size'   : 14}
        show_legend = True
        # Plot GP(x) + contours
        ax = plt.subplot(1, 1, 1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        ax = plot_gaussian_process(res, ax=ax,
                                noise_level=self.noise_level,
                                show_legend=show_legend, show_title=False,
                                show_next_point=False, show_acq_func=False)
        approx_x, approx_fn = expected_minimum(res)
        ax.legend(loc='upper right', prop=font)
        # ax.set_title(r"x* = %.4f" % (approx_x[0]))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(approx_x, approx_fn, 'b.', markersize=10, label=u'Minimum')
        # ax.set_xticks([0, 4, 8, 16, 32, 64, 128])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        # ax.set_yticks(**font)
        plt.xticks(**font)
        plt.yticks(**font)
        # current_x = plt.gca().get_xticks()
        # plt.gca().set_xticklabels(['{:,.0f}'.format(x * 140.0 + 10.0) for x in current_x])
        ax.set_ylabel("")
        ax.set_xlabel("")
    
    def plot_final_curve_step_i(self, res, step, ylim=None):
        font = {'family' : 'Times New Roman',
                'weight' : 'normal',
                'size'   : 14}
        show_legend = True
        # Plot GP(x) + contours
        ax = plt.subplot(1, 1, 1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        ax = plot_gaussian_process(res, ax=ax,
                                noise_level=self.noise_level,
                                show_legend=show_legend, show_title=False,
                                show_next_point=False, show_acq_func=False, n_calls=step)
        approx_x, approx_fn = expected_minimum(res)
        ax.legend(loc='upper right', prop=font)
        # ax.set_title(r"x* = %.4f" % (approx_x[0]))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(approx_x, approx_fn, 'b.', markersize=10, label=u'Minimum')
        # ax.set_xticks([0, 4, 8, 16, 32, 64, 128])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        # ax.set_yticks(**font)
        plt.xticks(**font)
        plt.yticks(**font)
        # current_x = plt.gca().get_xticks()
        # plt.gca().set_xticklabels(['{:,.0f}'.format(x * 140.0 + 10.0) for x in current_x])
        ax.set_ylabel("")
        ax.set_xlabel("")

    def opt_param_one_step(self):
        self.opt_gp = Optimizer([(100.0, 4000.0)], base_estimator="GP", n_initial_points=5,
                acq_optimizer="sampling", random_state=42)
        fig = plt.figure()
        fig.suptitle("Standard GP kernel")
        for i in range(20):
            next_x = self.opt_gp.ask()
            f_val = self.param_eval_fn(next_x)
            res = self.opt_gp.tell(next_x, f_val)
            if i >= 15:
                self.plot_optimizer(res, n_iter=i-15, max_iters=5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.plot()
        plt.show()
        plt.close()
        return res

    def gen_dataset(self, save_dir):
        if save_dir == '':
            self.save_dir = self.config['mpc_data']['folder']
        else:
            self.save_dir = save_dir
        os.system('mkdir -p ' + self.save_dir)
        save_yaml(self.config, os.path.join(self.save_dir, "config.yaml"))
        num_scene = self.config['mpc_data']['num_scene']
        self.record_mpc_data = self.config['mpc_data']['record_data']
        for scene_idx in range(num_scene):
            if self.config['mpc_data']['mode'] == 'random':
                print('generating scene:', scene_idx)
                if scene_idx % 2 == 1:
                    self.env.init_pos = 'rand_blob'
                    goal_row = np.random.randint(140, 500)
                    rand_char = chr(65 + np.random.randint(0, 26))
                    print('rand char:', rand_char)
                    self.subgoal, self.goal_img = gen_goal_shape(rand_char,
                                                                 h=self.screenHeight,
                                                                 w=self.screenWidth,
                                                                 font_name='helvetica')
                self.env.reset()
            elif self.config['mpc_data']['mode'] == 'same_init':
                self.env.init_pos = 'extra_small_half_spread'
                pos = np.load('init_pos/same_init_diff_goal_pos.npy')
                self.env.reset()
                self.env.set_positions(pos)
                if scene_idx % 2 == 0:
                    x = 360
                    y = 360
                    r = 150
                    self.subgoal, mask = gen_subgoal(x, y, r, h=self.screenHeight, w=self.screenWidth)
                    self.goal_img = (mask[..., None]*255).repeat(3, axis=-1).astype(np.uint8)
                else:
                    self.subgoal, self.goal_img = gen_goal_shape('K',
                                                                 h=self.screenHeight,
                                                                 w=self.screenWidth,
                                                                 font_name='helvetica')
            elif self.config['mpc_data']['mode'] == 'same_goal':
                if scene_idx % 2 == 0:
                    self.env.init_pos = 'center'
                    self.env.reset()
                else:
                    self.env.init_pos = 'center_init_2'
                    self.env.reset()
                x = 320
                y = 320
                r = 100
                self.subgoal, mask = gen_subgoal(x, y, r, h=self.screenHeight, w=self.screenWidth)
                self.goal_img = (mask[..., None]*255).repeat(3, axis=-1).astype(np.uint8)
            else:
                raise NotImplementedError
            
            self.funnel_dist = np.zeros_like(self.subgoal)
            self.last_pos = self.env.get_positions()

            for step_i in range(self.num_steps):
                self.pos = []
                self.eval_idx = 0
                self.save_dir_curr_step = os.path.join(self.save_dir, str(step_i + scene_idx * self.num_steps))
                os.system('mkdir -p ' + self.save_dir_curr_step)
                self.env.set_positions(self.last_pos)
                if self.record_mpc_data:
                    np.save(os.path.join(self.save_dir_curr_step, 'init_p.npy'), self.env.get_positions())
                # self.action_seq_mpc_init, self.action_label_seq_mpc_init = self.env.sample_action(self.n_mpc_per_model)
                self.action_seq_mpc_init = np.load('init_action/init_action_'+str(self.n_sample)+'.npy')[np.newaxis, ...] # [1, 50, 4]
                self.action_label_seq_mpc_init = np.zeros(1)
                # if self.config['mpc']['resample_p'] < 1.0:
                #     obs_cur = self.env.render()
                #     heatmap = (obs_cur[..., -1] < 0.599/0.8*self.global_scale).astype(np.float32) * self.subgoal
                #     heatmap = heatmap.reshape(-1)
                #     heatmap = softmax(heatmap)
                #     start_pixel_flat = np.random.choice(range(heatmap.shape[0]), p=heatmap)
                #     start_pixel = np.unravel_index(start_pixel_flat, self.subgoal.shape)
                #     target_area = (self.subgoal < 0.1).nonzero()
                #     target_pixel_idx = np.random.randint(target_area[0].shape[0])
                #     target_pixel = np.array([target_area[0][target_pixel_idx], target_area[1][target_pixel_idx]])
                #     start_action = self.env.pixel2action(start_pixel, w = self.subgoal.shape[0])
                #     target_action = self.env.pixel2action(target_pixel, w = self.subgoal.shape[0])
                #     self.action_seq_mpc_init[0] = np.concatenate((start_action, target_action), axis=0)
                raw_obs = self.env.render()
                save_obs(raw_obs, self.save_dir_curr_step, 0)
                cv2.imwrite(os.path.join(self.save_dir_curr_step, 'goal.png'), self.goal_img)
                init_state = ((raw_obs[..., -1] < 0.599/0.8 * self.global_scale) * 255)[..., None].repeat(3, axis=-1).astype(np.uint8)
                cv2.imwrite(os.path.join(self.save_dir_curr_step, 'init.png'), init_state)
                np.save(os.path.join(self.save_dir_curr_step, 'init_p.npy'), self.last_pos)
                
                init_rew = gt_rewards((raw_obs[..., -1] < 0.599/0.8 * self.global_scale).astype(np.float32), self.subgoal)
                
                ## optimize
                # kernel=1**2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=1e10)
                kernel=1**2 * Matern(length_scale=self.config['mpc_data']['gp']['length'], nu=self.config['mpc_data']['gp']['nu']) + WhiteKernel(noise_level=(self.config['mpc_data']['gp']['noise'] * init_rew) ** 2)
                base_estimator = GaussianProcessRegressor(kernel=kernel, normalize_y=True, noise="gaussian", n_restarts_optimizer=10)
                res = gp_minimize(self.param_eval_fn,
                                # [(10.0, 150.0)],      # the bounds on each dimension of x
                                # [(0.0, 1.0)],      # the bounds on each dimension of x
                                [Integer(2, 100)],
                                base_estimator=base_estimator, # the model
                                acq_func="EI",      # the acquisition function
                                n_calls=10,         # the number of evaluations of f
                                n_initial_points=0,  # the number of random initialization points
                                # x0=[[0.0], [20.0/140.0], [40.0/140.0], [90.0/140.0], [1.0]],  # initial points
                                x0=[[2], [25], [50], [75], [100]],  # initial points
                                # noise=0.5**2,       # the noise level (optional)
                                random_state=42)   # the random seed
                                    
                approx_x, approx_fn = expected_minimum(res)
                # dump(res, os.path.join(self.save_dir_curr_step, 'result.pkl'))
                # res = self.opt_param_one_step()

                print(res)
                # optimal_den = np.array([res.x[0]])
                optimal_den = np.array([approx_x])
                optimal_y = np.array([approx_fn])
                np.save(os.path.join(self.save_dir_curr_step, 'opt_den.npy'), optimal_den)
                np.save(os.path.join(self.save_dir_curr_step, 'opt_y.npy'), optimal_y)
                
                # save iteration info
                np.save(os.path.join(self.save_dir_curr_step, 'x_iters.npy'), np.array(res.x_iters))
                np.save(os.path.join(self.save_dir_curr_step, 'func_vals.npy'), np.array(res.func_vals))
                
                # plot the optimization process
                self.plot_optimizer(res, n_iter=0, max_iters=1)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(self.save_dir_curr_step, 'optimization_info.png'))
                plt.close()
                
                # plot the convergence
                plot_convergence(res)
                plt.savefig(os.path.join(self.save_dir_curr_step, 'convergence.png'))
                plt.close()

                # produce production-level image
                bg = cv2.imread('env/blank_bg.png')
                obs_wo_bg = rmbg(raw_obs[..., :3].astype(np.uint8), bg)[..., ::-1]

                overlay = cv2.addWeighted(obs_wo_bg, 0.5, self.goal_img, 0.5, 0)

                # self.plot_final_curve(res)
                # plt.savefig(os.path.join(self.save_dir_curr_step, 'final_curve.png'))
                # plt.close()
                for i in range(6):
                    self.plot_final_curve_step_i(res, i, [-5, 13])
                    plt.savefig(os.path.join(self.save_dir_curr_step, f'final_curve_step_{i}.png'))
                    plt.close()
                    final_curve_img = cv2.imread(os.path.join(self.save_dir_curr_step, f'final_curve_step_{i}.png'))

                    # resize final_curve_img to have the same height as bg
                    bg_h, bg_w, _ = bg.shape
                    plt_h, plt_w, _ = final_curve_img.shape
                    final_curve_img = cv2.resize(final_curve_img, (int(bg_h / plt_h * plt_w), bg_h))

                    # combine all three images
                    final_img = np.concatenate((obs_wo_bg, overlay, final_curve_img), axis=1)
                    cv2.imwrite(os.path.join(self.save_dir_curr_step, f'final_img_step_{i}.png'), final_img)

                self.last_pos = self.pos[np.random.randint(len(self.pos))]
            print()

def refine_res_plot(root_path, ylim, s_idx = 0, e_idx = 1, add_vid = True):
    if add_vid:
        vid = cv2.VideoWriter(os.path.join(root_path, ('refine_res_%d_%d.avi') % (s_idx, e_idx)), cv2.VideoWriter_fourcc(*'XVID'), 1, (2400, 720))
    gp_param_opt = GPParamOpt()
    for i in range(s_idx, e_idx):
        curr_root_path = os.path.join(root_path, str(i))
        res_path = os.path.join(curr_root_path, 'result.pkl')
        res = load(res_path)
        
        bg = cv2.imread('env/blank_bg.png')[..., ::-1]
        init = cv2.imread(os.path.join(curr_root_path, 'color_0.png'))
        goal_img = cv2.imread(os.path.join(curr_root_path, 'goal.png'))
        obs_wo_bg = rmbg(init, bg)

        overlay = cv2.addWeighted(obs_wo_bg, 0.5, goal_img, 0.5, 0)

        gp_param_opt.plot_final_curve(res, ylim=ylim)
        plt.savefig(os.path.join(curr_root_path, 'refined_curve.png'), dpi=500)
        plt.close()
        final_curve_img = cv2.imread(os.path.join(curr_root_path, 'refined_curve.png'))

        # resize final_curve_img to have the same height as bg
        bg_h, bg_w, _ = bg.shape
        plt_h, plt_w, _ = final_curve_img.shape
        final_curve_img = cv2.resize(final_curve_img, (int(bg_h / plt_h * plt_w), bg_h))

        # combine all three images
        final_img = np.concatenate((obs_wo_bg, overlay, final_curve_img), axis=1)
        cv2.imwrite(os.path.join(curr_root_path, 'refined_img.png'), final_img)
        if add_vid:
            vid.write(final_img)
    if add_vid:
        vid.release()

def plot_steps_vs_opt_x(root_path, s_idx, e_idx):
    gp_param_opt = GPParamOpt()
    opt_dens = np.zeros(e_idx - s_idx)
    for i in range(s_idx, e_idx):
        curr_root_path = os.path.join(root_path, str(i))
        opt_dens[i - s_idx] = np.load(os.path.join(curr_root_path, 'opt_den.npy'))
        # res_path = os.path.join(curr_root_path, 'result.pkl')
        # res = load(res_path)
        
        # bg = cv2.imread('env/blank_bg.png')[..., ::-1]
        # init = cv2.imread(os.path.join(curr_root_path, 'color_0.png'))
        # goal_img = cv2.imread(os.path.join(curr_root_path, 'goal.png'))
        # obs_wo_bg = rmbg(init, bg)

        # overlay = cv2.addWeighted(obs_wo_bg, 0.5, goal_img, 0.5, 0)

        # gp_param_opt.plot_final_curve(res, ylim=ylim)
        # plt.savefig(os.path.join(curr_root_path, 'refined_curve.png'))
        # plt.close()
        # final_curve_img = cv2.imread(os.path.join(curr_root_path, 'refined_curve.png'))

        # # resize final_curve_img to have the same height as bg
        # bg_h, bg_w, _ = bg.shape
        # plt_h, plt_w, _ = final_curve_img.shape
        # final_curve_img = cv2.resize(final_curve_img, (int(bg_h / plt_h * plt_w), bg_h))

        # # combine all three images
        # final_img = np.concatenate((obs_wo_bg, overlay, final_curve_img), axis=1)
        # cv2.imwrite(os.path.join(curr_root_path, 'refined_img.png'), final_img)
    plt.plot(np.arange(s_idx, e_idx), opt_dens)
    plt.savefig(os.path.join(root_path, 'steps_vs_opt_x.png'))
    

if __name__ == '__main__':
    # get variable from argparse
    parser = argparse.ArgumentParser(description='add optinal directory using command line')
    parser.add_argument('--dir', type=str, default='', help='directory to save results')
    args = parser.parse_args()
    gp_param_opt = GPParamOpt()
    gp_param_opt.gen_dataset(save_dir=args.dir)
    # refine_res_plot('data_gp_param_diff_init_same_goal_2', [-10.0, 12.5], s_idx=0, e_idx=1, add_vid=False)
    # plot_steps_vs_opt_x('data_gp_param_0130/all', 203, 213)
