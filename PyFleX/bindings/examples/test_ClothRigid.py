import os
import cv2
import scipy.misc
import numpy as np
import pyflex
import time
import torch


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


pyflex.init()

time_step = 300
dimx_cloth = 17
dimy_cloth = 17
height_cloth = 2
px_cloth = -0.75
py_cloth = -1.
pz_cloth = -0.75

dimx_rigid = 3
dimy_rigid = 3
dimz_rigid = 3
numx_rigid = 2
numy_rigid = 1
numz_rigid = 2

num_banana = 0
draw_points = 0

scene_params = np.array([
    dimx_cloth, dimy_cloth, height_cloth,
    px_cloth, py_cloth, pz_cloth,
    dimx_rigid, dimy_rigid, dimz_rigid,
    numx_rigid, numy_rigid, numz_rigid,
    num_banana, draw_points
])
pyflex.set_scene(12, scene_params, 0)

print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())

action = np.zeros(3)
act_scale = np.array([0.1, 0.1, 0.1])
dt = 1. / 60.

x = np.array([0., 1., 0.])
y = np.array([0., 0., 1.])

des_dir = 'test_ClothRigid'
os.system('mkdir -p ' + des_dir)

for i in range(time_step):

    positions = pyflex.get_positions().reshape(-1, 4)
    phases = pyflex.get_phases()

    cloth_start = dimx_rigid * dimy_rigid * dimz_rigid
    cloth_start *= numx_rigid * numy_rigid * numz_rigid

    center = positions[cloth_start, :3]

    if i == 0:
        offset = center

    action[0] += rand_float(-act_scale[0], act_scale[0]) - (center[0] - offset[0]) * act_scale[0]
    action[1] += rand_float(-act_scale[1], act_scale[1]) - (center[1] - offset[1]) * act_scale[1]
    action[2] += rand_float(-act_scale[2], act_scale[2]) - (center[2] - offset[2]) * act_scale[2]

    c = (x * i + y * (time_step - i)) / time_step
    pyflex.set_color(np.concatenate([[0], c]))

    c = (y * i + x * (time_step - i)) / time_step
    pyflex.set_color(np.concatenate([[1], c]))

    pyflex.step(action * dt, capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))

pyflex.clean()



fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(des_dir, 'out.avi'), fourcc, 60, (960, 720))
for i in range(time_step):
    img = scipy.misc.imread(os.path.join(des_dir, 'render_%d.tga' % i))
    out.write(img[..., :3][..., ::-1])
out.release()
