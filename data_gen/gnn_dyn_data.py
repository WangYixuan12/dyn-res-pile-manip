import os
import cv2
import pickle
import numpy as np
from env.flex_env import FlexEnv
import multiprocessing as mp
import time

# utils
from utils import load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed

config = load_yaml("config/data_gen/gnn_dyn.yaml")

data_dir = config['dataset']['folder']
n_episode = config['dataset']['n_episode']
n_timestep = config['dataset']['n_timestep']
action_dim = 4
num_worker = config['dataset']['num_worker']
obj = config['dataset']['obj']
wkspc_w = config['dataset']['wkspc_w']
if obj == 'ball':
    waitIter = 30
elif obj == 'coffee':
    waitIter = 100
elif obj == 'capsule':
    waitIter = 200
elif obj =='carrots':
    waitIter = 200
elif obj == 'coffee_capsule':
    waitIter = 200
else:
    raise ValueError("Unknown object: {}".format(obj))
global_scale = config['dataset']['global_scale']

os.system("mkdir -p " + data_dir)

def gen_data(info):
    base_epi = info["base_epi"]
    n_epi_per_worker = info["n_epi_per_worker"]
    thread_idx = info["thread_idx"]
    env = FlexEnv(config)
    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    idx_episode = base_epi
    # for idx_episode in range(base_epi, base_epi + n_epi_per_worker)
    while idx_episode < base_epi + n_epi_per_worker:
        env.reset()
        epi_dir = os.path.join(data_dir, '%d' % idx_episode)
        os.system("mkdir -p " + epi_dir)

        actions = np.zeros((n_timestep, action_dim))

        # execute the first action to perturb the system
        if obj == 'ball':
            action = None
            init_u = np.array([env.init_x + 1.2 * (np.random.randint(0, 2) - 0.6),
                            -env.init_z - 1.2 * (np.random.randint(0, 2) - 0.6),
                            env.init_x,
                            -env.init_z])
            action = init_u
            img = env.step(action)
            if img is None:
                print('rerun episode %d' % idx_episode)
                continue
        img = env.render()
        img[:, :, :3][img[:, :, -1] > 0.599/0.8 * global_scale] = np.ones(3) * 255.
        cv2.imwrite(os.path.join(epi_dir, '0_color.png'), img[:, :, :3][..., ::-1])
        cv2.imwrite(os.path.join(epi_dir, '0_depth.png'), (img[:, :, -1]*1000).astype(np.uint16))
        with open(os.path.join(epi_dir, '0_particles.npy'), 'wb') as f:
            np.save(f, env.get_positions())

        # s_centers = [[-0.3, 0.3], [0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.2, 0.0], [0.0, 0.2], [0.2, 0.0], [0.0, -0.2], [0.2, 0.2], [0.2, -0.2]]
        # e_centers = [[0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3], [0.3, 0.3], [0.2, 0.0], [0.0, -0.2], [-0.2, 0.0], [0.0, 0.2], [-0.2, -0.2], [-0.2, 0.2]]


        last_img = img.copy()
        valid = True
        for idx_timestep in range(n_timestep):
            # add pusher
            # s_2d = np.array(s_centers[idx_timestep])
            # e_2d = np.array(e_centers[idx_timestep])
            color_diff = 0
            while color_diff < 0.001:
                u = None
                u, _ = env.sample_action(1)
                u = u[0, 0]

                # step
                img = env.step(u)
                if img is None:
                    valid = False
                    print('rerun epsiode %d' % idx_episode)
                    break
                img[:, :, :3][img[:, :, -1] > 0.599/0.8 * global_scale] = np.ones(3) * 255.
                color_diff = np.mean(np.abs(img[:, :, :3] - last_img[:, :, :3]))
            if valid:
                cv2.imwrite(os.path.join(epi_dir, '%d_color.png' % (idx_timestep + 1)), img[:, :, :3][..., ::-1])
                cv2.imwrite(os.path.join(epi_dir, '%d_depth.png' % (idx_timestep + 1)), (img[:, :, -1]*1000).astype(np.uint16))
                with open(os.path.join(epi_dir, '%d_particles.npy' % (idx_timestep + 1)), 'wb') as f:
                    np.save(f, env.get_positions())
                actions[idx_timestep] = u
                last_img = img.copy()
            else:
                break
        if valid:
            idx_episode += 1

        with open(os.path.join(epi_dir, 'actions.p'), 'wb') as fp:
            pickle.dump(actions, fp)

    env.close()

# infos=[]
# for i in range(num_worker):
#     info = {
#         "base_epi": i*n_episode//num_worker,
#         "n_epi_per_worker": n_episode//num_worker,
#         "thread_idx": i
#     }
#     infos.append(info)

# pool = mp.Pool(processes=num_worker)
# pool.map(gen_data, infos)

info = {
    "base_epi": 0,
    "n_epi_per_worker": n_episode,
    "thread_idx": 1
}
gen_data(info)
