# Dynamic-Resolution Model Learning for Object Pile Manipulation

[Website](https://robopil.github.io/dyn-res-pile-manip/) | [Paper](https://arxiv.org/abs/2306.16700)

https://github.com/WangYixuan12/dyn-res-pile-manip/assets/32333199/60ad64d4-78f2-4654-bb2d-25599b9b66f4

Dynamic-Resolution Model Learning for Object Pile Manipulation  
[Yixuan Wang*](https://wangyixuan12.github.io/), [Yunzhu Li*](https://yunzhuli.github.io/), [Katherine Driggs-Campbell](https://krdc.web.illinois.edu/), [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li/), [Jiajun Wu](http://jiajunwu.com/)  
Robotics: Science and Systems, 2023.

## Citation

If you use this code for your research, please cite:

```
@INPROCEEDINGS{Wang-RSS-23, 
    AUTHOR    = {Yixuan Wang AND Yunzhu Li AND Katherine Driggs-Campbell AND Li Fei-Fei AND Jiajun Wu}, 
    TITLE     = {{Dynamic-Resolution Model Learning for Object Pile Manipulation}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2023}, 
    ADDRESS   = {Daegu, Republic of Korea}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2023.XIX.047} 
} 
@inproceedings{li2018learning,
    Title={Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids},
    Author={Li, Yunzhu and Wu, Jiajun and Tedrake, Russ and Tenenbaum, Joshua B and Torralba, Antonio},
    Booktitle = {ICLR},
    Year = {2019}
}
```

## Installation

### Prerequisite

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)
- Install [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/main/miniconda.html)

### Create conda environment
`conda env create -f env.yaml && conda activate dyn-res-pile-manip`

### Install PyFleX
Run `bash scripts/install_pyflex.sh`. You may need to `source ~/.bashrc` to `import PyFleX`.  
<details>
<summary>What does this script do?</summary>

We built our simulation using PyFleX. The original repository is [here](https://github.com/YunzhuLi/PyFleX). We modified it with additional features, such as depth rendering and headless rendering. We put the modified PyFleX in our repo. Please follow the commands below to install it.
```
docker pull xingyu/softgym
docker run \
    -v {PATH_TO_REPO}/PyFleX:/workspace/PyFleX \
    -v {PATH_TO_CONDA_ENV}:/workspace/anaconda \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash
```
For example, in my local machine, the command is
```
docker run \
    -v /home/yixuan/dyn-res-pile-manip/PyFleX:/workspace/PyFleX \
    -v /home/yixuan/miniconda3/envs/rss-release/:/workspace/anaconda \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash
```
After entering docker environment, run
```
export PATH="/workspace/anaconda/bin:$PATH"
cd /workspace/PyFleX
export PYFLEXROOT=${PWD}
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
cd bindings; mkdir build; cd build; /usr/bin/cmake ..; make -j
```
Add the following to `~/.bashrc`
```
export PYFLEXROOT=/home/yixuan/dyn-res-pile-manip/PyFleX # replace with your own path
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
```
</details>

### Load custom pybullet_data

Run `bash scripts/custom_robot_model.sh`  
<details>
<summary>What does this script do?</summary>

The script downloads `kinova.zip` and `franka_panda.zip` from the [release files](https://drive.google.com/drive/folders/1oKzLY3GzqPzv0FmGd_ipoLnz_rC7Ct5_) and extracts the custom end-effectors into the active Python environment's `pybullet_data` folder.
</details>

## Download data and model

The original release files are available in this [Google Drive folder](https://drive.google.com/drive/folders/1oKzLY3GzqPzv0FmGd_ipoLnz_rC7Ct5_). File IDs and SHA-256 checksums are recorded in [`scripts/download_assets.tsv`](scripts/download_assets.tsv).

Activate the project conda environment first. The scripts require `gdown` and Python; dataset and robot downloads also require `unzip`. They verify each downloaded file before installing it and store models and datasets under this repository's `data` folder, regardless of the working directory.

```bash
# Both pretrained checkpoints, ready for visualize_mpc.py:
bash scripts/download_model.sh

# Dynamics training data only (the default):
bash scripts/download_data.sh

# Resolution-regressor sanity-check data:
bash scripts/download_data.sh res_rgr_data_small

# Full resolution-regressor training data:
bash scripts/download_data.sh res_rgr_data

# All three datasets:
bash scripts/download_data.sh all
```

The archives total approximately 22 GB; allow additional space for extraction. Successfully extracted archives are removed. Rerunning a dataset or robot download overwrites matching extracted files; verified checkpoints and any verified archive left by an interrupted extraction are reused.

| Dataset | Extracted training path | Configured training sample count |
| --- | --- | --- |
| Dynamics | `data/gnn_dyn_data` | 2,000 episodes |
| Full resolution regressor | `data/res_rgr_data/all` | 1,433 |
| Small resolution regressor | `data/res_rgr_data_small` | 30 |

`config/train/res_rgr.yaml` defaults to the small dataset. To train on the full dataset, set `train_res_cls.data_root` to `data/res_rgr_data/all` and `train_res_cls.num_data` to `1433`. The original full and small archives contain 1,434 and 31 numbered directories respectively; the counts above preserve the released training configurations.

## Task in sim

Run `python visualize_mpc.py`. You will see our robot system push the spreaded object pile into an I-shape pile.

<details>
<summary>Change initial configuration and task</summary>

Initial configurations and task are specified in `config/mpc/config.yaml`. `init_pos` provides some options of object pile initial states. Task specification can be changed in `task`. 
</details>

## Train GNN dynamics model

Run `python -m train.train_gnn_dyn`. It will save the trained model in `data/gnn_dyn_model`.

## Train resolution regressor

Run `python -m train.train_res_rgr`. It will save the trained model in `data/res_rgr_model`.

## Data generation

### Generate data for dynamics model

Run `python -m data_gen.gnn_dyn_data`. It will save the generated data in `data/gnn_dyn_data_custom`.

### Generate data for resolution regressor

Run `python -m data_gen.res_rgr_data`. It will save the generated data in `data/res_rgr_data_custom`.

<details>
<summary>Different data generation modes</summary>

You could specify the data generation mode in `config/data_gen/res_rgr.yaml` by changing `mpc_data/mode`. If you generate resolution regressor training data in `random` mode, it will synthesize random initial configurations and goals. If you want to re-produce Fig. 4 (a) in the paper, you could change it to `same_init` mode. For Fig. 4 (b) in the paper, you could change it to `same_goal` mode. Due to stochasity, it may not produce exactly the same result as in the paper.
</details>
