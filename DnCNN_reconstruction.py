import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import tensorboardX

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from Unet_util import *
from utils import prepare_sub_folder
from DnCNN import *


# data_loader = get_data_loader("features_and_data/fbp_train.npy", "/data/bowen/SparseReconstruction/3d-ct-full-dose/cropped_train.npy")

data_loader = get_data_loader('features_and_data/fbp_train_lidc.npy', '/data/bowen/SparseReconstruction/lidc_train.npy')
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']
# model_name = "DnCNN_6262022"
model_name = "DnCNN_7052022"
cudnn.benchmark = True
output_folder = '/data/bowen/SparseReconstruction/3d-ct-full-dose/models'
# train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
model = DnCNN(channels = 1)
model.cuda()
if config['optimizer'] == 'Adam':
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
else:
    NotImplementedError

# Setup loss function
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    raise NotImplementedError
    
model.train()

cur_iter = 1
for e in tqdm(range(max_iter)):
    total_train_loss = 0
    for it, (fbp_raw, gt) in enumerate(data_loader):
        
        fbp_raw = fbp_raw.transpose(1,3).float()
        gt = gt.transpose(1,3).float()
        fbp_raw = fbp_raw.cuda()
        gt = gt.cuda()
        train_data = (fbp_raw, gt)
        train_output = model(train_data[0])
        train_loss = loss_fn(train_output, train_data[1])
        optim.zero_grad()
        total_train_loss += train_loss
        train_loss.backward()
        optim.step()
       
    total_train_loss = total_train_loss.cpu().detach().numpy()/3480
        
    print("train loss at ",cur_iter, " iteration is ", total_train_loss)
    cur_iter += 1
        

model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (max_iter + 1))
torch.save({'net': model.state_dict(), \
    'opt': optim.state_dict(), \
    }, model_name)


