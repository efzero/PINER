import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import tensorboardX
from torch.autograd import grad
from torch_radon import Radon
from prior_utils import *

import numpy as np
from tqdm import tqdm



import numpy as np
from tqdm import tqdm

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader, ct_parallel_project_2d_batch


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--pretrain', action='store_true', help="load pretrained model weights")
parser.add_argument('--slice', type=int, default=None, help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']
recon_name = config['recon_name']
prior_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test/'
reg = config['reg']
############testing noise###############
ind = int(config['prior_index'])
############testing noise###############
# noise = np.load("/data/bowen/SparseReconstruction/3d-ct-full-dose/noise.npy")[ind:(ind+1),:,:,0] #1,25,256,1
# noise = noise*0  ####PSNR
# noise = noise*256*1


#####################################load noise, need to change for different testing settings#####################################
# noise = np.load('features_and_data/simulated_noise_large.npy')  ###for LDCT DnCNN

noise = np.load('features_and_data/lowdose_noise_unet.npy')


noise = noise.swapaxes(1,2)
noise = noise[ind:(ind+1), :,:] #1, 25, 256
print(np.mean(noise**2), noise.shape)
# noise = np.load("notebooks/reduced_noise.npy")*256
# noise = noise*1  ####PSNR
noise = noise*256*1
########################################################################################################################





#####################################prior file, need to change for different testing settings#####################################
# prior = np.load("/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon/prior" + str(ind) + ".npy")

# prior = np.load('/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test/cnn_recon_test' + str(ind) + '.npy')
# prior = np.load('/data/bowen/SparseReconstruction/3d-ct-full-dose/adaptive_prior/img'+str(ind) + 'npy.npy')
# prior = np.load('/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test2/cnn_recon_test' + str(ind) + '.npy')
# prior = np.load('/data/bowen/SparseReconstruction/3d-ct-full-dose/dncnn_recon_test/dncnn_prior' + str(ind) + '.npy') ##for DnCNN LDCT recon

# prior = np.load(prior_path + 'unet_lowdose_recon' + str(ind)  + '.npy') ###for normal low dose unet
# prior = np.load(prior_path + 'dncnn_lowdose_recon' + str(ind)  + '.npy') ###for normal low dose dncnn
prior = np.load(prior_path + 'unet_robust_recon_continuous' + str(ind) + '.npy')
prior = np.clip(crop_img(prior),0,1)
prior = prior.reshape((1,256,256,1))

########################################################################################################################



# Setup optimizer

# Setup loss function
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    NotImplementedError


    
# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], -1, train=True, batch_size=config['batch_size'])

proj_noise = torch.from_numpy(noise).to(dtype = torch.float32).cuda(3)
prior = torch.from_numpy(prior).to(dtype = torch.float32).cuda(3)
x = torch.zeros_like(prior, requires_grad=True).cuda(3)

if config['optimizer'] == 'Adam':
    optim = torch.optim.Adam(params = [x], lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
else:
    NotImplementedError
    



angles = np.linspace(0, np.pi, config['num_projs'], endpoint=False)
radon = Radon(256, angles, clip_to_circle=True)


for it, (grid, image) in enumerate(data_loader):
    print(type(grid), type(image))
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda(3)  # [bs, h, w, 2], [0, 1]
    image = image.cuda(3)  # [bs, h, w, c], [0, 1]
    
    
    print(grid.shape, image.shape)

    # 2D parallel projection geometry

    projs = radon.forward(image[:,:,:,0])
#     Data loading

    test_data = (grid, image)
    train_data = (grid, projs)
    
    
    # Train model
    for iterations in range(max_iter):
        
        optim.zero_grad()

        train_projs = radon.forward(x[:,:,:,0])
        
        # print(noise_level)
        

        train_loss = 0
# #         ###########################TEST SPARSITY CONSTRAINT METHOD#############################################
        if reg == 'lasso': 
# #             print('using lasso')
        
            res = x-prior
            res_dx = res[:,:,1:,0] - res[:,:,:-1,0]
            res_dy = res[:,1:,:,0] - res[:,:-1,:,0]
            x_dx = x[:,:,1:,0] - x[:,:,:-1,0]
            x_dy = x[:,1:,:,0] - x[:,:-1,:,0]

            mse_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1])
            
    
            train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1]) +5e-5*256**2*6*(torch.mean(torch.abs(res_dx))*1.4 + torch.mean(torch.abs(res_dy))*1.4 + torch.mean(torch.abs(x_dx))*.6 + torch.mean(torch.abs(x_dy))*.6)
        
        
        elif reg == 'res-lasso':    
            
            recovered = x + prior
            x_dx = recovered[:,:,1:,0] - recovered[:,:,:-1,0]
            x_dy = recovered[:,1:,:,0] - recovered[:,:-1,:,0]
            train_loss = 0.5*loss_fn(train_projs + proj_noise, train_data[1] - prior_proj) + 5e-5*256**2*.75*torch.mean(torch.abs(x)) + 5e-5*256**2*1.5*(torch.mean(torch.abs(x_dx)) + torch.mean(torch.abs(x_dy)))
        
#             train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1]) +5e-5*256**2*6*torch.mean(torch.abs(res)) + \
#     5e-5*256**2*12*(torch.mean(torch.abs(x_dx)) + torch.mean(torch.abs(x_dy)))
    
        elif reg == 'lasso_ata':
            train_loss = 0.5*loss_fn(radon.backprojection(train_projs+proj_noise), radon.backprojection(train_data[1])) + 5e-5*torch.mean(torch.abs(train_output)**2)
            
        elif reg == 'ridge_ata':
            train_loss = 0.5*loss_fn(radon.backprojection(train_projs+proj_noise), radon.backprojection(train_data[1])) + 5e-3*torch.mean(train_output**2)
            
        elif reg == 'priorcnn':
            train_loss = 0.5*loss_fn(train_projs + proj_noise, train_data[1]) + 5e-5*256**2*3*torch.mean(torch.abs(prior - train_output))
        else:
            train_loss = 0.5*loss_fn(radon.backprojection(train_projs+proj_noise), radon.backprojection(train_data[1]))

            
        # mse_l = mse_loss.detach().cpu().numpy()/256/256
        # if mse_l < noise_level**2/2*.80:
        #     break
        if iterations%1000 == 0:
            # print(mse_l)
            print(train_loss.detach().cpu().numpy(), "trainloss")
        train_loss.backward()
        optim.step()
        

np.save(recon_name, x.detach().cpu().numpy())


# np.save(recon_name, (x+prior).detach().cpu().numpy())
    
torch.cuda.empty_cache()
