import os
import argparse
import random

from tqdm import tqdm
import numpy as np

import time
import wandb
import pytorch_fid

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from dataset import get_loader
from model import Generator, Discriminator


class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

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


class Trainer(object):
    '''Procedes the training and computes the loss'''
    def __init__(self, cfg):
        
        self.cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.img_size = cfg.img_size
        self.img_dir = cfg.img_dir
        self.test_dir = cfg.test_dir
        self.d_ckpt_dir = cfg.d_ckpt_dir
        self.g_ckpt_dir = cfg.g_ckpt_dir
        self.sample_dir = cfg.sample_dir
        self.mode = cfg.mode

        # training
        self.batch_size = cfg.batch_size
        self.latent_dim = cfg.latent_dim
        self.max_g_dim = cfg.max_g_dim
        self.max_d_dim = cfg.max_d_dim
        self.max_iter = cfg.max_iter

        #optimizer
        self.g_optimizer = cfg.g_optimizer
        self.d_optimizer = cfg.d_optimizer
        # learning rate
        self.lr = cfg.lr
        self.scheduler_name = cfg.scheduler_name
        # Adam
        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        # SGD
        self.momentum = cfg.momentum
        # ReduceLROnPlateau
        self.factor = cfg.factor
        self.patience = cfg.patience
        self.eps = cfg.eps
        # CosineAnnealingLR
        self.T_max = cfg.T_max
        self.min_lr = cfg.min_lr

        # print/sample/checkpoint
        self.print_every = cfg.print_every
        self.sample_every = cfg.sample_every
        self.ckpt_every = cfg.ckpt_every

        # test
        self.test_batch_size = cfg.test_batch_size
        
        # build model
        self.make_model()

    def latent_code(self):
        '''Make latent code'''
        noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
        return noise

    def criterion(self):
        '''Set loss function'''
        loss = nn.BCELoss()
        return loss

    def make_model(self):
        '''Make Generator and Discriminator and set optimizer'''
        self.model_G = Generator(self.latent_dim, self.max_g_dim)
        self.model_D = Discriminator(self.max_d_dim)

        self.set_g_optim('Adam', self.model_G)
        self.set_d_optim('Adam', self.model_D)

    def set_g_optim(self, optim, generator):
        '''For Generator optimizer'''
        if optim == 'Adam':
            self.g_optim = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        elif optim == 'SGD':
            self.g_optim = torch.optim.SGD(generator.parameters(), lr=self.lr, momentum=self.momentum)

    def set_d_optim(self, optim, discriminator):
        '''For Discriminator optimizer'''
        if optim == 'Adam':
            self.d_optim = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        
        elif optim == 'SGD':
            self.d_optim = torch.optim.SGD(discriminator.parameters(), lr=self.lr, momentum=self.momentum)

    def get_scheduler(self, optimizer):
        if self.scheduler_name=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.factor, patience=self.patience, verbose=True, eps=self.eps)
        elif self.scheduler_name=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.min_lr, last_epoch=-1)
        
        return scheduler

    def make_labels(self, real_label=1, fake_label=0):
        '''label_real -> 1, label_fake -> 0'''
        label_real = torch.full((self.batch_size,), fill_value=real_label)
        label_fake = torch.full((self.batch_size,), fill_value=fake_label)

        return label_real, label_fake

    def train(self):
        '''Training'''
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_d = AverageMeter()
        losses_g = AverageMeter()

        model_G = self.model_G
        model_D = self.model_D

        model_G = model_G.to(self.device)
        model_D = model_D.to(self.device)

        start = end = time.time()
        loader = get_loader(self.img_dir, self.img_size, self.batch_size, self.mode)
        data_loader = iter(loader)
        pbar = tqdm(range(self.max_iter))
        num_iter = 0

        fixed_noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)

        for i in pbar:
            data_time.update(time.time() - end)

            # prevent stopiteration
            try:
                img = next(data_loader)

            except (OSError, StopIteration):
                data_loader = iter(loader)
                img = next(data_loader)
            img = img.to(self.device)

            #criterion
            criterion = self.criterion()

            # noise
            noise = self.latent_code()

            # zero_grad
            model_D.zero_grad()

            # loss_D_real
            out_D_real = model_D(img).view(-1)
            out_D_real = out_D_real.type(torch.FloatTensor).to(self.device)
            label_real, label_fake = self.make_labels()
            label_real = label_real.type(torch.FloatTensor).to(self.device)
            loss_D_real = criterion(out_D_real, label_real)

            # loss_G_real
            out_G = model_G(noise)
            out_D_fake = model_D(out_G.detach()).view(-1)
            out_D_fake = out_D_fake.type(torch.FloatTensor).to(self.device)
            label_fake = label_fake.type(torch.FloatTensor).to(self.device)
            loss_D_fake = criterion(out_D_fake, label_fake)

            # loss_D
            loss_D = loss_D_fake + loss_D_real

            losses_d.update(loss_D.item(), self.batch_size)

            # D backpropagation
            loss_D.backward()
            self.d_optim.step()

            # zero_grad
            model_G.zero_grad()

            out_G_real = model_D(out_G).view(-1)
            loss_G = criterion(out_G_real, label_real)

            losses_g.update(loss_G.item(), self.batch_size)
            
            # G backpropagation
            loss_G.backward()
            self.g_optim.step()

            num_iter += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.print_every == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (i, self.max_iter, loss_D.item(), loss_G.item()))

            if ((i+1) % self.ckpt_every == 0) or (i==self.max_iter):
                save_name_g = os.path.join(self.g_ckpt_dir, 'it{:06d}-G.pt'.format(num_iter))
                save_name_d = os.path.join(self.d_ckpt_dir, 'it{:06d}-D.pt'.format(num_iter))
                print('Saving models in each directory')
                torch.save({'model_G_state_dict': self.model_G.state_dict()}, save_name_g)
                torch.save({'model_D_state_dict': self.model_D.state_dict()}, save_name_d)

            if (i+1) % self.sample_every == 0:
                with torch.no_grad():
                    sample = model_G(fixed_noise)
                    file_path = os.path.join(self.sample_dir, '{}.jpg'.format(i+1))
                save_image(sample, file_path)

    def test(self):
        noise = torch.randn(self.test_batch_size, self.latent_dim, 1, 1, device=self.device)
        for i in os.listdir('/home/hyunin/GAN_practice/assets/g_ckpt'):
            load_ckpt = torch.load(os.path.join('/home/hyunin/GAN_practice/assets/g_ckpt', i))
            model_t = Generator(100, 1024)
            model_t.load_state_dict(load_ckpt['model_G_state_dict'])
            model_t.to(self.device) 
            model_t.eval()

            file_path = os.path.join(self.test_dir, 'test-{}.jpg'.format(i[2:7]))
            test_sample = model_t(noise)
            save_image(test_sample, file_path)
            
        

        
