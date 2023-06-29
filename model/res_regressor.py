import os
import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as models

class MPCResCls(nn.Module):
    
    def __init__(self, config):
        
        super(MPCResCls, self).__init__()
        
        self.config = config
        
        self.state_h = config['train_res_cls']['state_h']
        self.state_w = config['train_res_cls']['state_w']
        self.res_dim = config['train_res_cls']['res_dim']
        input_ch = 6
        
        # a design similar to pix2pix
        self.model = nn.Sequential(
            nn.Conv2d(input_ch, 64, 4, 2, 1),   # 32 x 32
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, 4, 2, 1),        # 16 x 16
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, 4, 2, 1),       # 8 x 8
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, 4, 2, 1),       # 4 x 4
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, 4, 2, 1),       # 4 x 4
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 6),
            # nn.Softmax(dim=1)
        )
        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.state_h * self.state_w * input_ch, 2048),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(64, 6),
        # )
        
    def forward(self, x):
        return self.model(x)
    
    def infer_param(self, init_img, goal_img):
        # init: binary numpy of shape (H, W)
        # goal: binary numpy of shape (H, W)
        assert init_img.shape == goal_img.shape
        # plt.subplot(1, 2, 1)
        # plt.imshow(init_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(goal_img)
        # plt.show()
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
        input_img_tensor = torch.from_numpy(input_img[None, ...]).float().cuda()
        output_img_tensor = self.forward(input_img_tensor)
        res_idx = output_img_tensor.argmax(dim=1).item()
        res = [4, 8, 16, 32, 64, 128][res_idx]
        return res

class MPCResRgrNoPool(nn.Module):
    
    def __init__(self, config):
        
        super(MPCResRgrNoPool, self).__init__()
        
        self.config = config
        
        self.state_h = config['train_res_cls']['state_h']
        self.state_w = config['train_res_cls']['state_w']
        self.res_dim = config['train_res_cls']['res_dim']
        input_ch = 6
        
        # a design similar to pix2pix
        self.model = nn.Sequential(
            nn.Conv2d(input_ch, 64, 4, 2, 1),   # 32 x 32
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, 4, 2, 1),        # 16 x 16
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, 4, 2, 1),       # 8 x 8
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, 4, 2, 1),       # 4 x 4
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, 4, 2, 1),       # 4 x 4
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        return self.model(x)
    
    def infer_param(self, init_img, goal_img):
        # init: binary numpy of shape (H, W)
        # goal: binary numpy of shape (H, W)
        assert init_img.shape == goal_img.shape
        # plt.subplot(1, 2, 1)
        # plt.imshow(init_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(goal_img)
        # plt.show()
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
        input_img_tensor = torch.from_numpy(input_img[None, ...]).float().cuda()
        output_img_tensor = self.forward(input_img_tensor)
        particle_num = (output_img_tensor).item() # * 140.0 + 10.0
        return int(particle_num)
