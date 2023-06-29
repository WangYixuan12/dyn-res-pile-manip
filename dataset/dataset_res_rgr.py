import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from scipy.special import softmax

from utils import preprocess_action_segment, preprocess_action_repeat, load_yaml
import skopt

class DatasetResRgr(Dataset):
    
    def __init__(self, data_dir, config, phase):
        self.config = config
        
        self.num_data = config['train_res_cls']['num_data']
        self.global_scale = config['dataset']['global_scale']

        train_valid_ratio = config['train_res_cls']['train_valid_ratio']
        
        n_train = int(self.num_data * train_valid_ratio)
        n_valid = self.num_data - n_train

        if phase == 'train':
            self.epi_st_idx = 0
            self.n_episode = n_train
        elif phase == 'valid':
            self.epi_st_idx = n_train
            self.n_episode = n_valid
        else:
            raise AssertionError("Unknown phase %s" % phase)

        self.data_dir = data_dir        

        self.state_h = config['train_res_cls']['state_h']
        self.state_w = config['train_res_cls']['state_w']
        self.model_type = config['train_res_cls']['model_type']

    def __len__(self):
        return self.n_episode
    
    def rewards2score(self, rewards):
        # rewards: (n_models, n_mpc, n_repeat)
        # return: (n_models, )
        final_rewards = rewards[:, -1 , :]
        rewards_reduction = rewards[:, 0, :] - rewards[:, -1, :]
        rewards_reduction = np.mean(rewards_reduction, axis=1)
        rewards_reduction = rewards_reduction / 1e5
        return softmax(rewards_reduction)

    def rewards2target(self, rewards):
        # rewards: (n_models, n_mpc, n_repeat)
        # return: (n_models, )
        final_rewards = rewards[:, -1 , :]
        rewards_reduction = rewards[:, 0, :] - rewards[:, -1, :]
        rewards_reduction = np.mean(rewards_reduction, axis=1)
        return np.argmax(rewards_reduction)

    def __getitem__(self, idx):
        init_path = os.path.join(self.data_dir, '%d/init.png' % (idx + self.epi_st_idx))
        init_img = cv2.imread(init_path)
        goal_path = os.path.join(self.data_dir, '%d/goal.png' % (idx + self.epi_st_idx))
        goal_img = cv2.imread(goal_path)
        
        init_img = init_img[..., 0] / 255.0
        goal_img = goal_img[..., 0] / 255.0
        
        init_img_dist = cv2.distanceTransform((1- init_img).astype(np.uint8), cv2.DIST_L2, 5) / (init_img.shape[0])
        goal_img_dist = cv2.distanceTransform((1- goal_img).astype(np.uint8), cv2.DIST_L2, 5) / (goal_img.shape[0])
        
        init_exclude_goal = np.logical_and(init_img, 1-goal_img)
        goal_exclude_init = np.logical_and(goal_img, 1-init_img)
        
        init_img = cv2.resize(init_img, (self.state_w, self.state_h), interpolation=cv2.INTER_AREA)
        goal_img = cv2.resize(goal_img, (self.state_w, self.state_h), interpolation=cv2.INTER_AREA)
        init_img_dist = cv2.resize(init_img_dist, (self.state_w, self.state_h), interpolation=cv2.INTER_AREA)
        goal_img_dist = cv2.resize(goal_img_dist, (self.state_w, self.state_h), interpolation=cv2.INTER_AREA)
        init_exclude_goal = cv2.resize(init_exclude_goal.astype(np.float32), (self.state_w, self.state_h), interpolation=cv2.INTER_AREA)
        goal_exclude_init = cv2.resize(goal_exclude_init.astype(np.float32), (self.state_w, self.state_h), interpolation=cv2.INTER_AREA)
        
        input_img = np.concatenate([init_img[None, ...],
                                    goal_img[None, ...],
                                    init_img_dist[None, ...],
                                    goal_img_dist[None, ...],
                                    init_exclude_goal[None, ...],
                                    goal_exclude_init[None, ...]], axis=0)
        input_img_tensor = torch.from_numpy(input_img).float()
        
        # plt.subplot(1, 3, 1)
        # plt.imshow(init_img)
        # plt.subplot(1, 3, 2)
        # plt.imshow(goal_img)
        # plt.subplot(1, 3, 3)
        # plt.plot(scores)
        # plt.show()
        # plt.close()
        
        if self.model_type == 'classifier':
            optimal_den_path = os.path.join(self.data_dir, '%d/opt_den.npy' % (idx + self.epi_st_idx))
            optimal_den = np.load(optimal_den_path)
            # rewards = np.load(rewards_path)
            
            # scores = self.rewards2score(rewards)
            # target = self.rewards2target(rewards)
            # scores_tensor = torch.from_numpy(scores).float()
            scores_tensor = torch.ones(1).float()
            resolutions = np.array([4, 8, 16, 32, 64, 128])
            target = (resolutions == optimal_den[0]).nonzero()[0][0]
            target_tensor = torch.from_numpy(np.array([target])).long()
            return {'input_img': input_img_tensor, 'scores': scores_tensor, 'target': target_tensor}
        elif self.model_type == 'regressor':
            optimal_den_path = os.path.join(self.data_dir, '%d/opt_den.npy' % (idx + self.epi_st_idx))
            optimal_den = np.load(optimal_den_path)
            optimal_den_tensor = torch.from_numpy(optimal_den).float() #  * 140.0 + 10.0 # between 0 and 1; scale to real value by * 140 + 10
            # optimal_den_tensor = optimal_den_tensor / 4000.0
            
            # conf_path = os.path.join(self.data_dir, '%d/conf.npy' % (idx + self.epi_st_idx))
            # conf = np.load(conf_path)
            # conf = np.ones(1)
            opt_y_path = os.path.join(self.data_dir, '%d/opt_y.npy' % (idx + self.epi_st_idx))
            opt_y = np.load(opt_y_path)
            conf = np.minimum(np.exp(- opt_y - 1.0), 1.0)
            conf_tensor = torch.from_numpy(conf).float()
                
            return {'input_img': input_img_tensor, 'optimal_den': optimal_den_tensor, 'conf': conf_tensor}
        else:
            raise AssertionError("Unknown model type %s" % self.model_type)

def test_dataset():
    config = load_yaml('config.yaml')

    dataset = DatasetResRgr(config['train_res_cls']['data_root'], config, 'train')
    data = dataset[np.random.randint(0, len(dataset))]

def compute_dataset_conf():
    # compute the confidence of the optimal density
    config = load_yaml('config.yaml')

    data_dir = config['train_res_cls']['data_root']
    length = config['train_res_cls']['num_data']

    for i in range(length):
        optimal_den_path = os.path.join(data_dir, '%d/opt_den.npy' % (i))
        optimal_den = np.load(optimal_den_path)
        res_path = os.path.join(data_dir, '%d/result.pkl' % (i))
        res = skopt.load(res_path)
        test_x = np.array([100., 500., 1000., 2000., 3000., 4000.])
        closest_idx = np.argmin(np.abs(test_x - optimal_den))
        test_x[closest_idx] = optimal_den
        model_x = (test_x - 100.) / 3900. # normalize to [0, 1]
        model_pred = res.models[-1].predict(model_x[:, np.newaxis])
        conf = np.array([softmax(model_pred / -5e4)[closest_idx]])
        conf_path = os.path.join(data_dir, '%d/conf.npy' % (i))
        np.save(conf_path, conf)

def compute_dataset_opt_y():
    # compute the confidence of the optimal density
    config = load_yaml('config.yaml')

    data_dir = config['train_res_cls']['data_root']
    length = config['train_res_cls']['num_data']

    for i in range(length):
        optimal_den_path = os.path.join(data_dir, '%d/opt_den.npy' % (i))
        optimal_den = np.load(optimal_den_path)
        res_path = os.path.join(data_dir, '%d/result.pkl' % (i))
        res = skopt.load(res_path)
        model_pred = res.models[-1].predict(optimal_den)
        opt_y_path = os.path.join(data_dir, '%d/opt_y.npy' % (i))
        np.save(opt_y_path, model_pred)

def viz_dataset():
    # visualize the distribution of optimal density
    config = load_yaml('config.yaml')
    log_dir = 'mpc_res_dataset_viz'
    os.system('mkdir -p %s' % log_dir)

    dataset = DatasetResRgr(config['train_res_cls']['data_root'], config, 'train')
    densities = np.zeros((len(dataset)))
    conf = np.zeros((len(dataset)))
    for i in range(len(dataset)):
        data = dataset[i]
        densities[i] = data['optimal_den'].item()
        conf[i] = data['conf'].item()
    plt.hist(densities, bins=10)
    plt.savefig(os.path.join(log_dir, 'density_hist.png'))
    plt.close()
    
    plt.hist(conf, bins=10)
    plt.savefig(os.path.join(log_dir, 'conf_hist.png'))
    plt.close()
    
    # select samples for optimal density around [100, 500, 1000, 2000, 3000, 4000]
    num_samples = 3
    # viz_density = [100, 500, 1000, 2000, 3000, 4000]
    viz_density = [10, 30, 50, 100, 150]
    for i, den in enumerate(viz_density):
        is_range = np.logical_and(densities > den - 50, densities < den + 50)
        if np.sum(is_range) == 0:
            continue
        for j in range(num_samples):
            idx = np.random.choice(np.where(is_range)[0])
            print('idx:', idx)
            data = dataset[idx]
            init_img = data['input_img'][0].numpy()
            goal_img = data['input_img'][1].numpy()
            print('conf:', data['conf'].item())
            plt.subplot(1, 3, 1)
            plt.imshow(init_img)
            plt.subplot(1, 3, 2)
            plt.imshow(goal_img)
            plt.subplot(1, 3, 3)
            plt.hist(densities, bins=10)
            plt.axvline(data['optimal_den'].numpy() * 4000., color='r')
            plt.savefig(os.path.join(log_dir, '%d_%d.png' % (idx, den)))
            os.system('cp %s %s' % (os.path.join(config['train_res_cls']['data_root'], '%d/optimization_info.png' % (idx + dataset.epi_st_idx)), os.path.join(log_dir, '%d_%d_opt.png' % (idx, den))))
            plt.close()

def viz_dataset_for_cls():
    # visualize the distribution of optimal density
    config = load_yaml('config.yaml')
    log_dir = 'mpc_res_dataset_viz'
    os.system('mkdir -p %s' % log_dir)

    dataset = DatasetResRgr(config['train_res_cls']['data_root'], config, 'train')
    densities = np.zeros((len(dataset)))
    for i in range(len(dataset)):
        data = dataset[i]
        densities[i] = data['target'].item()
    plt.hist(densities, bins=10)
    plt.show()
    # plt.savefig(os.path.join(log_dir, 'density_hist.png'))
    plt.close()

    # select samples for optimal density around [100, 500, 1000, 2000, 3000, 4000]
    num_samples = 3
    viz_target = [0, 1, 2, 3, 4]
    for i, den in enumerate(viz_target):
        print('den:', den)
        is_range = (densities == den)
        if np.sum(is_range) == 0:
            continue
        for j in range(num_samples):
            idx = np.random.choice(np.where(is_range)[0])
            print('idx:', idx)
            data = dataset[idx]
            init_img = data['input_img'][0].numpy()
            goal_img = data['input_img'][1].numpy()
            plt.subplot(1, 2, 1)
            plt.imshow(init_img)
            plt.subplot(1, 2, 2)
            plt.imshow(goal_img)
            plt.show()
            # plt.subplot(1, 3, 3)
            # plt.hist(densities, bins=10)
            # plt.axvline(data['optimal_den'].numpy() * 4000., color='r')
            # plt.savefig(os.path.join(log_dir, '%d_%d.png' % (idx, den)))
            # os.system('cp %s %s' % (os.path.join(config['train_res_cls']['data_root'], '%d/optimization_info.png' % (idx + dataset.epi_st_idx)), os.path.join(log_dir, '%d_%d_opt.png' % (idx, den))))
            plt.close()

if __name__ == '__main__':
    # test_dataset()
    viz_dataset()
    # viz_dataset_for_cls()
    # compute_dataset_conf()
    # compute_dataset_opt_y()
