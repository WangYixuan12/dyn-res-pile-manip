import os
import time
import random
import matplotlib.pyplot as plt

import numpy as np
import cv2

# utils
from utils import get_current_YYYY_MM_DD_hh_mm_ss_ms, save_yaml, load_yaml

# dynamics
from utils import rand_int, count_trainable_parameters, Tee, AverageMeter, get_lr, to_np, set_seed
from model.res_regressor import MPCResCls, MPCResRgrNoPool
from dataset.dataset_res_rgr import DatasetResRgr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, DataLoader

import scipy

def train_res_cls(
    config,
    train_dir,
    data_dir,
    model,
    global_iteration,):

    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train', 'valid']:
        print("Loading data for %s" % phase)
        datasets[phase] = DatasetResRgr(
            data_dir,
            config,
            phase=phase)

        dataloaders[phase] = DataLoader(
            datasets[phase],
            batch_size=config['train_res_cls']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train_res_cls']['num_worker'],
            drop_last=True)

        data_n_batches[phase] = len(dataloaders[phase])

    use_gpu = torch.cuda.is_available()

    # shape:
    state_h = config['train_res_cls']['state_h']
    state_w = config['train_res_cls']['state_w']
    model_type = config['train_res_cls']['model_type']

    # criterion
    KLDivLoss = nn.KLDivLoss(log_target=True)
    MSELoss = nn.MSELoss(reduction='none')
    HuberLoss = nn.HuberLoss(delta=200./4000., reduction='none')
    CELoss = nn.CrossEntropyLoss()

    # optimizer
    params = model.parameters()
    lr = float(config['train_res_cls']['lr'])
    optimizer = optim.Adam(params, lr=lr, betas=(config['train_res_cls']['adam_beta1'], 0.999))

    # setup scheduler
    sc = config['train_res_cls']['lr_scheduler']
    scheduler = None

    if config['train_res_cls']['lr_scheduler']['enabled']:
        if config['train_res_cls']['lr_scheduler']['type'] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=sc['factor'],
                patience=sc['patience'],
                threshold_mode=sc['threshold_mode'],
                cooldown= sc['cooldown'],
                verbose=True)
        elif config['train_res_cls']['lr_scheduler']['type'] == "StepLR":
            step_size = config['train_res_cls']['lr_scheduler']['step_size']
            gamma = config['train_res_cls']['lr_scheduler']['gamma']
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError("unknown scheduler type: %s" %(config['train_res_cls']['lr_scheduler']['type']))

    if use_gpu:
        print("using gpu")
        model = model.cuda()


    best_valid_loss = np.inf
    counters = {'train': 0, 'valid': 0}

    try:
        for epoch in range(config['train_res_cls']['n_epoch']):
            phases = ['train', 'valid']

            for phase in phases:
                meter_loss = AverageMeter()
                model.train(phase == 'train')

                # bar = ProgressBar(max_value=data_n_batches[phase])
                loader = dataloaders[phase]

                for i, data in enumerate(loader):

                    global_iteration += 1
                    counters[phase] += 1

                    with torch.set_grad_enabled(phase == 'train'):
                        input_img = data['input_img']
                        if model_type == 'classifier':
                            scores = data['scores']
                            target = data['target']
                        elif model_type == 'regressor':
                            scores = data['optimal_den'][:, 0]
                            confs = data['conf']
                        else:
                            raise ValueError("unknown model type: %s" % model_type)
                        B = scores.shape[0]

                        if use_gpu:
                            input_img = input_img.cuda()
                            scores = scores.cuda()
                            if model_type == 'regressor':
                                confs = confs.cuda()
                            elif model_type == 'classifier':
                                target = target.cuda()
                            else:
                                raise ValueError("unknown model type: %s" % model_type)

                        # visualize input image and ground truth
                        # input_img_np = input_img[0].detach().cpu().numpy()
                        # print('corresponding score:', scores[0].detach().cpu().numpy() * 140.0 + 10.0)
                        # plt.subplot(2, 2, 1)
                        # plt.imshow(input_img_np[0])
                        # plt.colorbar()
                        # plt.subplot(2, 2, 2)
                        # plt.imshow(input_img_np[1])
                        # plt.colorbar()
                        # plt.subplot(2, 2, 3)
                        # plt.imshow(input_img_np[2])
                        # plt.colorbar()
                        # plt.subplot(2, 2, 4)
                        # plt.imshow(input_img_np[3])
                        # plt.colorbar()
                        # plt.show()
                        # plt.close()
                        
                        output_scores = model(input_img)

                        if model_type == 'classifier':
                            # loss_kl = KLDivLoss(output_scores, scores) + KLDivLoss(scores, output_scores)
                            # loss_kl = KLDivLoss(scores, output_scores)
                            loss_ce = CELoss(output_scores, target[:, 0])
                        elif model_type == 'regressor':
                            loss_mse = (MSELoss(output_scores, scores) * confs).mean()
                            # loss_mse = (HuberLoss(output_scores, scores) * confs).mean()
                        loss_reg = 0.
                        n_param = 0.
                        for ii, W in enumerate(list(model.model.parameters())):
                            if ii % 2 == 0:
                                # print(i, W.size())
                                loss_reg += W.norm(1)
                                n_param += W.numel()
                        loss_reg /= n_param

                        if model_type == 'classifier':
                            loss = loss_ce + loss_reg * float(config['train_res_cls']['lam_reg'])
                        elif model_type == 'regressor':
                            loss = loss_mse + loss_reg * float(config['train_res_cls']['lam_reg'])
                        meter_loss.update(loss.item(), B)


                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                    if i % config['train_res_cls']['log_per_iter'] == 0:
                        log = '%s %d [%d/%d][%d/%d] LR: %.6f' % (
                            phase, global_iteration, epoch, config['train_res_cls']['n_epoch'], i, data_n_batches[phase],
                            get_lr(optimizer))
                        log += ', loss: %.6f' % (loss.item())
                        if model_type == 'classifier':
                            log += ', ce: %.6f' % (loss_ce.item())
                        elif model_type == 'regressor':
                            log += ', mse: %.6f' % (loss_mse.item())
                        log += ', reg: %.6f' % (loss_reg.item())

                        print(log)

                    if phase == 'train' and global_iteration % config['train_res_cls']['ckp_per_iter'] == 0:
                        save_model(model, '%s/net_dy_iter_%d' % (train_dir, global_iteration))

                log = '%s %d [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                    phase, global_iteration, epoch, config['train_res_cls']['n_epoch'], meter_loss.avg, best_valid_loss)
                print(log)


                if phase == "train":
                    if (scheduler is not None) and (config['train_res_cls']['lr_scheduler']['type'] == "StepLR"):
                        scheduler.step()

                if phase == 'valid':
                    if (scheduler is not None) and (config['train_res_cls']['lr_scheduler']['type'] == "ReduceLROnPlateau"):
                        scheduler.step(meter_loss.avg)

                    if meter_loss.avg < best_valid_loss:
                        best_valid_loss = meter_loss.avg
                        save_model(model, '%s/net_best_dy' % (train_dir))

    except KeyboardInterrupt:
        # save network if we have a keyboard interrupt
        save_model(model, '%s/net_dy_iter_%d_keyboard_interrupt' % (train_dir, global_iteration))


    return model, global_iteration

def save_model(model, save_base_path):
    # save both the model in binary form, and also the state dict
    torch.save(model.state_dict(), save_base_path + "_state_dict.pth")
    torch.save(model, save_base_path + "_model.pth")

def test_pred_overfit(model_folder, iter_num=-1):
    config = load_yaml('config.yaml')
    model_root = 'data/res_rgr_model'
    model_folder = os.path.join(model_root, model_folder)
    model_type = config['train_res_cls']['model_type']
    os.system('mkdir -p regressor_viz') 
    if model_type == 'classifier':
        model = MPCResCls(config)
    elif model_type == 'regressor':
        model = MPCResRgrNoPool(config)
        # model = MPCResRgrMLP(config)
    if iter_num == -1:
        model.load_state_dict(torch.load(os.path.join(model_folder, 'net_best_dy_state_dict.pth')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_folder, 'net_dy_iter_%d_state_dict.pth' % iter_num)))
    dataset = DatasetResRgr(config['train_res_cls']['data_root'], config, 'valid')
    for i in range(len(dataset)):
        # if i == 0:
        #     idx = 80
        # else:
        #     idx = np.random.randint(0, len(dataset))
        idx = i
        print('idx', idx)
        data = dataset[idx]
        pred_score = model(data['input_img'].unsqueeze(0))
        if model_type == 'classifier':
            plt.subplot(2, 2, 1)
            plt.imshow(data['input_img'].cpu().numpy()[0])
            plt.subplot(2, 2, 2)
            plt.imshow(data['input_img'].cpu().numpy()[1])
            plt.subplot(2, 2, 3)
            plt.plot(data['scores'].cpu().numpy())
            plt.subplot(2, 2, 4)
            plt.plot(pred_score.detach().cpu().numpy()[0])
        elif model_type == 'regressor':
            fig=plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(data['input_img'].cpu().numpy()[0])
            plt.subplot(2, 2, 2)
            plt.imshow(data['input_img'].cpu().numpy()[1])
            plt.subplot(2, 2, 3)
            plt.imshow(data['input_img'].cpu().numpy()[2])
            plt.subplot(2, 2, 4)
            plt.imshow(data['input_img'].cpu().numpy()[3])
            pred_particle_num = (pred_score * 140.0 + 10.0).detach().cpu().numpy()[0]
            gt_particle_num = (data['optimal_den'] * 140.0 + 10.0).detach().cpu().numpy()
            conf_num = data['conf'].detach().cpu().numpy()
            fig.suptitle('pred: %.2f, gt: %.2f, conf: %.2f' % (pred_particle_num, gt_particle_num, conf_num))
            plt.savefig('regressor_viz/%d.png' % idx)
            print('pred', pred_score * 140.0 + 10.0)
            print('gt', data['optimal_den'] * 140.0 + 10.0)
            print('conf', data['conf'])
            print('MSE', np.mean((pred_score.detach().cpu().numpy()[0] - ((data['optimal_den'].cpu().numpy()))) ** 2))
        # plt.show()
        plt.close()

def train():
    config = load_yaml('config/train/res_rgr.yaml')
    data_dir = config['train_res_cls']['data_root']
    model_type = config['train_res_cls']['model_type']
    train_root = 'data/res_rgr_model'
    train_dir = os.path.join(train_root, get_current_YYYY_MM_DD_hh_mm_ss_ms())
    os.system('mkdir -p ' + train_root)
    os.system('mkdir -p ' + train_dir)

    # set random seed for reproduction
    set_seed(config['train_res_cls']['random_seed'])

    # save the config
    save_yaml(config, os.path.join(train_dir, "config.yaml"))

    print(config)

    if model_type == 'classifier':
        model = MPCResCls(config)
    elif model_type == 'regressor':
        model = MPCResRgrNoPool(config)
        # model = MPCResRgrMLP(config)
    else:
        raise NotImplementedError
    print(model)

    global_iteration = 0

    ### optimize the dynamics model
    model, global_iteration = train_res_cls(
        config,
        train_dir,
        data_dir,
        model,
        global_iteration,)

if __name__ == '__main__':
    train()
