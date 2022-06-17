import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.utils as vutils

from model import VQVAE, MyVQVAE



class Trainer(object):
    def __init__(self, dataset, val_data, params):
        ### Misc ###
        self.device = params.device

        ### Make Dirs ###
        self.log_dir = params.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        ### load/save params
        if params.load_params:
            with open(os.path.join(params.log_dir, 'params.pkl'), 'rb') as file:
                params = pickle.load(file)
        else:
            with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)

        self.p = params

        ### Make Models ###
        self.model = VQVAE(self.p).to(self.p.device)

        if self.p.ngpu > 1:
            self.model = nn.DataParallel(self.model,device_ids=list(range(self.p.ngpu)))

        self.opt = optim.Adam(self.model.parameters(), lr=self.p.lr, betas=(0., 0.9))
        self.scaler = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.val_data = DataLoader(val_data, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.loss = nn.MSELoss()

        ### Prep Training
        self.losses = []
        self.val_losses = []
        self.fid = []
        self.fid_epoch = []

    def inf_train_gen(self):
        while True:
            for data in self.generator_train:
                yield data
        
    def log_train(self, step, data, rec):
        with torch.no_grad():
            data = torch.nan_to_num(data, nan=0.0, posinf=1, neginf=-1)
            rec = torch.nan_to_num(rec, nan=0.0, posinf=1, neginf=-1)
            self.fid.append(
                FID.fid(
                    torch.reshape(rec.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(data.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1)
                    )
                )
        
        rec, com = self.losses[-1]
        v_rec, v_com = self.val_losses[-1]

        print('[%d/%d]\tReconstruction Loss: %.4f\tCommitment Loss: %.4f\t Val: %.4f|%.4f\tFID %.4f'
                    % (step, self.p.niters, rec, com, v_rec, v_com, self.fid[-1]))

    def log_interpolation(self, step, data, rec):
        torchvision.utils.save_image(
            vutils.make_grid(torch.reshape(data, (-1,1,128,128)), padding=2, normalize=True)
            , os.path.join(self.images_dir, f'in_{step}.png'))
        torchvision.utils.save_image(
            vutils.make_grid(torch.reshape(rec, (-1,1,128,128)), padding=2, normalize=True)
            , os.path.join(self.images_dir, f'rec_{step}.png'))

    def start_from_checkpoint(self):
        step = 0
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']

            self.opt.load_state_dict(state_dict['opt'])
            self.model.load_state_dict(state_dict['model'])

            self.losses = state_dict['loss']
            self.val_lossses = state_dict['val']
            self.fid_epoch = state_dict['fid']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        torch.save({
        'step': step,
        'model': self.model.state_dict(),
        'opt': self.opt.state_dict(),
        'loss': self.losses,
        'val': self.val_losses,
        'fid': self.fid_epoch,
        }, os.path.join(self.models_dir, 'checkpoint.pt'))

    def log(self, step, data, rec):
        if step % self.p.steps_per_log == 0:
            self.val_step()
            self.log_train(step, data, rec)

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(step, data, rec)

    def log_final(self, step, data, rec):
        self.log_train(step, data, rec)
        self.log_interpolation(step, data, rec)
        self.save_checkpoint(step)

    def val_step(self):
        with torch.no_grad():
            with autocast():
                l = [[],[]]
                for x in self.val_data:
                    x = x.unsqueeze(1).to(self.p.device)
                    rec, (commitment_loss, _, _) = self.model(x)
                    rec = torch.tanh(rec)

                    rec_loss = self.loss(rec, x)
                    commitment_loss = 0# sum(commitment_loss).mean()

                    l[0].append(rec_loss.item())
                    l[1].append(commitment_loss)#.item())


        self.val_losses.append(tuple(np.mean(l, axis=1)))

    def train_step(self, x):
        for p in self.model.parameters():
            p.requires_grad = True

        self.model.zero_grad()
        
        rec, (commitment_loss, q,_) = self.model(x)
        rec = torch.tanh(rec)

        rec_loss = self.loss(rec, x)
        commitment_loss = sum(commitment_loss).mean()

        loss = rec_loss + commitment_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        for p in self.model.parameters():
            p.requires_grad = False
        return rec.detach(), rec_loss.item(), commitment_loss.item()

    def train(self):
        step_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)
        gen = self.inf_train_gen()

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            data = next(gen).unsqueeze(1)
            rec, rec_loss, commitment_loss = self.train_step(data.to(self.p.device))
            self.losses.append((rec_loss, commitment_loss))

            self.log(i, data, rec)
            if i%100 == 0 and i>0:
                self.fid_epoch.append(np.array(self.fid).mean())
                self.fid = []
                self.save_checkpoint(i)
        
        self.log_final(i, data, rec)
        print('...Done')
