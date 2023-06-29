import os
import cv2
import sys
import numpy as np
import torch
import random
import datetime
import time
import yaml
from PIL import Image, ImageOps, ImageEnhance

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import torch
from torch.autograd import Variable

from dgl.geometry import farthest_point_sampler
import open3d as o3d

from PIL import Image, ImageEnhance

def rect_from_coord(uxi, uyi, uxf, uyf, bar_width):
    # transform into angular coordinates
    theta = np.arctan2(uyf - uyi, uxf - uxi)
    length = np.linalg.norm(np.array([uxf - uxi, uyf - uyi]), ord=2)

    theta0 = theta - np.pi / 2.

    v = np.array([bar_width / 2.0 * np.cos(theta0),\
                  bar_width / 2.0 * np.sin(theta0)])

    st = np.array([uxi, uyi])
    ed = np.array([uxf, uyf])

    st0 = st + v
    st1 = st - v
    ed0 = ed + v
    ed1 = ed - v

    return st0, st1, ed1, ed0


def check_side(a, b):
    return a[0] * b[1] - b[0] * a[1]


def check_within_rect(x, y, rect):
    p = np.array([x, y])
    p0, p1, p2, p3 = rect

    side0 = check_side(p - p0, p1 - p0)
    side1 = check_side(p - p1, p2 - p1)
    side2 = check_side(p - p2, p3 - p2)
    side3 = check_side(p - p3, p0 - p3)

    if side0 >= 0 and side1 >= 0 and side2 >= 0 and side3 >= 0:
        return True
    elif side0 <= 0 and side1 <= 0 and side2 <= 0 and side3 <= 0:
        return True
    else:
        return False


def preprocess_action_segment(act):
    # generate the action frame to illustrate the pushing segment
    # each position in the pushing segment contains the offset to the end

    width = 32
    height = 32
    bar_width = 32. / 500 * 80

    act = act + 0.5

    act_frame = np.zeros((2, height, width))

    uxi = float(width) * act[0]
    uyi = float(height) * act[1]
    uxf = float(width) * act[2]
    uyf = float(height) * act[3]

    st = np.array([uxi, uyi])
    ed = np.array([uxf, uyf])

    rect = rect_from_coord(uxi, uyi, uxf, uyf, bar_width)

    direct = np.array([uxf - uxi, uyf - uyi])
    direct = direct / np.linalg.norm(direct, ord=2)

    for i in range(height):
        for j in range(width):
            x = j + 0.5
            y = (height - i) - 0.5
            cur = np.array([x, y])

            if check_within_rect(x, y, rect):
                to_ed = ed - cur
                to_ed = to_ed / np.linalg.norm(to_ed, ord=2)
                angle = np.arccos(np.dot(direct, to_ed))

                length = np.linalg.norm(ed - cur, ord=2) * np.cos(angle)
                offset = length * direct

                act_frame[:, i, j] = offset / np.array([width, height])

    '''
    for i in range(height):
        print(act_frame[0, i, :].tolist())
    print()
    for i in range(height):
        print(act_frame[1, i, :].tolist())

    time.sleep(1000)
    '''

    return act_frame.reshape(-1)



def preprocess_action_repeat(act, width=32, height=32):
    # generate the action frame by appending index with action
    # each position contains the coordinate and the action
    # act: 4 / 6 / ...
    act_dim = act.shape[0]
    act_frame = np.zeros((act_dim+2, height, width))

    act_frame[2:] = np.tile(act.reshape(-1, 1, 1), (1, height, width))
    width_1d = (np.arange(width) + 0.5) / width - 0.5
    height_1d = (height - np.arange(height) - 0.5) / height - 0.5
    act_frame[0] = np.tile(width_1d.reshape(1, 1, -1), (1, height, 1))
    act_frame[1] = np.tile(height_1d.reshape(1, -1, 1), (1, 1, width))

    return act_frame.reshape(-1)



def preprocess_action_repeat_tensor(act, width=32, height=32, pos_enc=None):
    # generate the action frame by appending index with action
    # each position contains the coordinate and the action
    # act: B x 4
    assert type(act) == torch.Tensor

    B, act_dim = act.size()

    act_frame = torch.zeros((B, 2 + act_dim, height, width), dtype=torch.float32, device=act.device)
    if pos_enc is not None:
        act_frame[:, :2] = pos_enc.repeat(B, 1, 1, 1)
    else:
        act_frame[:, 0] = ((torch.arange(width).reshape(1, 1, -1) + 0.5) / width - 0.5).repeat(B, height, 1)
        act_frame[:, 1] = ((height - torch.arange(height).reshape(1, -1, 1) - 0.5) / height - 0.5).repeat(B, 1, width)
    act_frame[:, 2:] = act.reshape(B, act_dim, 1, 1).repeat(1, 1, height, width)

    # act_frame: B x (6 * height * width)
    return act_frame.view(B, -1).cuda()




def get_current_YYYY_MM_DD_hh_mm_ss_ms():
    """
    Returns a string identifying the current:
    - year, month, day, hour, minute, second

    Using this format:

    YYYY-MM-DD-hh-mm-ss

    For example:

    2018-04-07-19-02-50

    Note: this function will always return strings of the same length.

    :return: current time formatted as a string
    :rtype: string

    """

    now = datetime.datetime.now()
    string =  "%0.4d-%0.2d-%0.2d-%0.2d-%0.2d-%0.2d-%0.6d" % (now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
    return string


def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))


def save_yaml(data, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def calc_dis(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def norm(x, p=2):
    return np.power(np.sum(x ** p), 1. / p)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_non_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor), requires_grad=requires_grad)


def to_np(x):
    return x.detach().cpu().numpy()


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt(
        (std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


'''
image utils
'''

def resize(img, size, interpolation=Image.BILINEAR):

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))


def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):

    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


'''
record utils
'''

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def drawRotatedRect(img, s, e, width=1):
    # img: (h, w, 3) numpy arra
    # s: (x, y) tuple
    # e: (x, y) tuple
    # color will change from (255, 0, 0) to (255, 255, 0)
    l = int(np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) + 1)
    theta = np.arctan2(e[1] - s[1], e[0] - s[0])
    theta_ortho = theta + np.pi / 2
    for i in range(l):
        color = (255, int(255 * i / l), 0)
        x = int(s[0] + (e[0] - s[0]) * i / l)
        y = int(s[1] + (e[1] - s[1]) * i / l)
        img = cv2.line(img.copy(), (int(x - 0.5 * width * np.cos(theta_ortho)), int(y - 0.5 * width * np.sin(theta_ortho))), 
                    (int(x + 0.5 * width * np.cos(theta_ortho)), int(y + 0.5 * width * np.sin(theta_ortho))), color, 1)
    return img

def drawPushing(img, s, e, width):
    # img: (h, w, 3) numpy arra
    # s: (x, y) tuple
    # e: (x, y) tuple
    # color will change from (255, 0, 0) to (255, 255, 0)
    l = int(np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) + 1)
    theta = np.arctan2(e[1] - s[1], e[0] - s[0])
    theta_ortho = theta + np.pi / 2
    img = cv2.line(img.copy(), (int(s[0] - 0.5 * width * np.cos(theta_ortho)), int(s[1] - 0.5 * width * np.sin(theta_ortho))), 
                (int(s[0] + 0.5 * width * np.cos(theta_ortho)), int(s[1] + 0.5 * width * np.sin(theta_ortho))), (255,99,71), 5)
    img = cv2.line(img.copy(), (int(e[0] - 0.5 * width * np.cos(theta_ortho)), int(e[1] - 0.5 * width * np.sin(theta_ortho))),
                (int(e[0] + 0.5 * width * np.cos(theta_ortho)), int(e[1] + 0.5 * width * np.sin(theta_ortho))), (255,99,71), 5)
    img = cv2.arrowedLine(img.copy(), (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), (255,99,71), 5)
    return img

def findClosestPoint(pcd, point):
    # pcd: (n, 3) numpy array
    # point: (3,) numpy array
    dist = np.linalg.norm(pcd - point[None, :], axis=1)
    return np.argmin(dist)

def fps(pcd, particle_num, init_idx=-1):
    # pcd: (n, 3) numpy array
    # pcd_fps: (self.particle_num, 3) numpy array
    pcd_tensor = torch.from_numpy(pcd).float()[None, ...]
    if init_idx == -1:
        # init_idx = findClosestPoint(pcd, pcd.mean(axis=0))
        pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num)[0]
    else:
        pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num, init_idx)[0]
    pcd_fps_tensor = pcd_tensor[0, pcd_fps_idx_tensor]
    pcd_fps = pcd_fps_tensor.numpy()
    dist = np.linalg.norm(pcd[:, None] - pcd_fps[None, :], axis=-1)
    dist = dist.min(axis=1)
    return pcd_fps, dist.max()

def fps_rad(pcd, radius):
    # pcd: (n, 3) numpy array
    # pcd_fps: (-1, 3) numpy array
    # radius: float
    rand_idx = np.random.randint(pcd.shape[0])
    pcd_fps_lst = [pcd[rand_idx]]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while dist.max() > radius:
        pcd_fps_lst.append(pcd[dist.argmax()])
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    return pcd_fps

def fps_np(pcd, particle_num, init_idx=-1):
    # pcd: (n, c) numpy array
    # pcd_fps: (particle_num, c) numpy array
    # radius: float
    if init_idx == -1:
        rand_idx = np.random.randint(pcd.shape[0])
        # rand_idx = findClosestPoint(pcd, pcd.mean(axis=0))
    else:
        rand_idx = init_idx
    pcd_fps_lst = [pcd[rand_idx]]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while len(pcd_fps_lst) < particle_num:
        pcd_fps_lst.append(pcd[dist.argmax()])
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    return pcd_fps, dist.max()
    
def recenter(pcd, sampled_pcd, r = 0.02):
    # pcd: (n, 3) numpy array
    # sampled_pcd: (self.partcile_num, 3) numpy array
    # recentering around a local point cloud
    particle_num = sampled_pcd.shape[0]
    dist = np.linalg.norm(pcd[:, None, :] - sampled_pcd[None, :, :], axis=2) # (n, self.particle_num)
    recenter_sampled_pcd = np.zeros_like(sampled_pcd)
    for i in range(particle_num):
        recenter_sampled_pcd[i] = pcd[dist[:, i] < r].mean(axis=0)
    return recenter_sampled_pcd

def opengl2cam(pcd, cam_extrinsic, global_scale):
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    opencv_T_world = np.matmul(np.linalg.inv(cam_extrinsic), opencv_T_opengl)
    # print('opencv_T_world inverse', np.linalg.inv(opencv_T_world))
    cam = np.matmul(np.linalg.inv(opencv_T_world), np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1).T).T[:, :3] / global_scale
    # print('cam', cam)
    # print()
    return cam

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def pcd2pix(pcd, cam_params, offset=(0, 0)):
    # pcd: (n, 3)
    # pix: (n, 2), (row, col)
    # offset: offset in the image space
    fx, fy, cx, cy = cam_params
    pix = np.zeros((pcd.shape[0], 2))
    try:
        pix[:, 1] = pcd[:, 0] * fx / pcd[:, 2] + cx
        pix[:, 0] = pcd[:, 1] * fy / pcd[:, 2] + cy
        pix[:, 0] += offset[0]
        pix[:, 1] += offset[1]
    except:
        print('pcd', pcd)
        exit(1)
    return pix.astype(np.int32)

def rmbg(img, bg):
    # img: (h, w, 3)
    # bg: (h, w, 3)
    assert img.shape == bg.shape
    assert img.dtype == np.uint8
    img_diff = np.abs(img.astype(np.int32) - bg.astype(np.int32)).sum(axis=2)
    img[img_diff < 1e-3] = np.ones(3, dtype=np.uint8) * 255
    return img

def downsample_pcd(pcd, voxel_size):
    # pcd: (n, 3) numpy array
    # downpcd: (m, 3)
    
    # convert numpy array to open3d point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

    downpcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    downpcd = np.asarray(downpcd_o3d.points)

    return downpcd

def gt_rewards(mask, subgoal):
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask)
    # plt.subplot(1, 2, 2)
    # plt.imshow(subgoal)
    # plt.show()
    subgoal_mask = subgoal < 0.5
    obj_dist = cv2.distanceTransform(1 - mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.sum(mask * subgoal) / mask.sum() + np.sum(obj_dist * subgoal_mask) / subgoal_mask.sum()

def gt_rewards_norm_by_sum(mask, subgoal):
    subgoal_mask = subgoal < 0.5
    obj_dist = cv2.distanceTransform(1 - mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.sum(mask * subgoal) / subgoal.sum() + np.sum(obj_dist * subgoal_mask) / obj_dist.sum()

dodger_blue_RGB = (30, 144, 255)
dodger_blue_BGR = (255, 144, 30)
tomato_RGB = (255, 99, 71)
tomato_BGR = (71, 99, 255)

def gen_goal_shape(name, h, w, font_name='helvetica_thin'):
    root_dir = f'env/target_shapes/{font_name}'
    shape_path = os.path.join(root_dir, 'helvetica_' + name + '.npy')
    goal = np.load(shape_path)
    goal = cv2.resize(goal, (w, h), interpolation=cv2.INTER_AREA)
    goal = (goal <= 0.5).astype(np.uint8)
    goal_dist = np.minimum(cv2.distanceTransform(1-goal, cv2.DIST_L2, 5), 1e4)
    goal_img = (goal * 255)[..., None].repeat(3, axis=-1).astype(np.uint8)
    # plt.subplot(1,2,1)
    # plt.imshow(goal)
    # plt.subplot(1,2,2)
    # plt.imshow(goal_dist)
    # plt.show()
    return goal_dist, goal_img

def gen_ch_goal(name, h, w):
    root_dir = 'env/target_shapes/720_ch'
    shape_path = os.path.join(root_dir, name + '.npy')
    goal = np.load(shape_path)
    goal = cv2.resize(goal, (w, h), interpolation=cv2.INTER_AREA)
    goal = (goal <= 0.5).astype(np.uint8)
    goal_dist = cv2.distanceTransform(1-goal, cv2.DIST_L2, 5)
    goal_img = (goal * 255)[..., None].repeat(3, axis=-1).astype(np.uint8)
    # plt.subplot(1,2,1)
    # plt.imshow(goal)
    # plt.subplot(1,2,2)
    # plt.imshow(goal_dist)
    # plt.show()
    return goal_dist, goal_img

def gen_subgoal(c_row, c_col, r, h = 64, w = 64):
    mask = np.zeros((h, w))
    grid = np.mgrid[0:h, 0:w]
    grid[0] = grid[0] - c_row
    grid[1] = grid[1] - c_col
    dist = np.sqrt(np.sum(grid**2, axis=0))
    mask[dist < r] = 1
    subgoal = np.minimum(cv2.distanceTransform((1-mask).astype(np.uint8), cv2.DIST_L2, 5), 1e4)
    return subgoal, mask

def lighten_img(img, factor=1.2):
    # img: assuming an RGB image
    assert img.dtype == np.uint8
    assert img.shape[2] == 3
    cv2.imwrite('tmp_1.png', img)
    img = Image.open('tmp_1.png').convert("RGB")
    img_enhancer = ImageEnhance.Brightness(img)
    enhanced_output = img_enhancer.enhance(factor)
    enhanced_output.save("tmp_2.png")
    color_lighten_img = cv2.imread('tmp_2.png')
    os.system('rm tmp_1.png tmp_2.png')
    return color_lighten_img

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d
