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
from Unet import *




#################################################set up the loading directory of data loader#################################################

# data_loader = get_data_loader("fbp_train.npy", '/data/bowen/SparseReconstruction/3d-ct-full-dose/train_multiorgan_slices_unbiased.npy')
# data_loader = get_data_loader("fbp_train.npy", "/data/bowen/SparseReconstruction/3d-ct-full-dose/cropped_train.npy")
# data_loader = get_data_loader('features_and_data/fbp_train_lidc.npy', '/data/bowen/SparseReconstruction/lidc_train.npy') ###train normal LIDC
data_loader = get_data_loader('features_and_data/fbp_train_lidc_robust.npy', '/data/bowen/SparseReconstruction/lidc_train.npy') ###train robust LIDC
# data_loader = get_data_loader("features_and_data/fbp_train_robust.npy", "/data/bowen/SparseReconstruction/3d-ct-full-dose/cropped_train.npy")  ###train robust LDCT

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']
# model_name = "Unet_7032022"
# model_name = "Unet_8212022_robust"
model_name = "Unet_8212022_robust_lidc"

cudnn.benchmark = True

# # Setup output folder
# output_folder = os.path.splitext(os.path.basename(opts.config))[0]
output_folder = '/data/bowen/SparseReconstruction/3d-ct-full-dose/models'


train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


# # Setup input encoder:

# # Setup model
model = Unet(use_dropout=True, num_down=3)
print(model)
model.cuda(2)
# model.train()

# # Setup optimizer
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


# # Setup data loader
# print('Load image: {}'.format(config['img_path']))
# data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], -1, train=True, batch_size=config['batch_size'])

cur_iter = 1
for e in tqdm(range(max_iter)):
    model.train()
    total_train_loss = 0
    for it, (fbp_raw, gt) in enumerate(data_loader):
        
        fbp_raw = fbp_raw.transpose(1,3).float()
        gt = gt.transpose(1,3).float()
        fbp_raw = fbp_raw.cuda(2)
        gt = gt.cuda(2)
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
        
    
#     # Input coordinates (x,y) grid and target image
#     grid = grid.cuda()  # [bs, h, w, 2], [0, 1]
#     image = image.cuda()  # [bs, h, w, c], [0, 1]
    
#     # Data loading 
#     # Change training inputs for downsampling image
#     test_data = (grid, image)
#     train_data = (grid, image)
#     print(test_data[1].cpu().shape, train_data[1].cpu().shape)

#     torchvision.utils.save_image(test_data[1].cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "test.png"))
#     torchvision.utils.save_image(train_data[1].cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "train.png"))

#     # Train model
#     for iterations in range(max_iter):
#         model.train()
#         optim.zero_grad()

#         train_embedding = encoder.embedding(train_data[0])  # [B, H, W, embedding*2]
#         train_output = model(train_embedding)  # [B, H, W, 3]
#         train_loss = 0.5 * loss_fn(train_output, train_data[1])

#         train_loss.backward()
#         optim.step()

#         # Compute training psnr
#         if (iterations + 1) % config['log_iter'] == 0:
#             train_psnr = -10 * torch.log10(2 * train_loss).item()
#             train_loss = train_loss.item()

#             train_writer.add_scalar('train_loss', train_loss, iterations + 1)
#             train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
#             print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss, train_psnr))

#         # Compute testing psnr
#         if (iterations + 1) % config['val_iter'] == 0:
#             model.eval()
#             with torch.no_grad():
#                 test_embedding = encoder.embedding(test_data[0])
#                 test_output = model(test_embedding)

#                 test_loss = 0.5 * loss_fn(test_output, test_data[1])
#                 test_psnr = - 10 * torch.log10(2 * test_loss).item()
#                 test_loss = test_loss.item()

#             train_writer.add_scalar('test_loss', test_loss, iterations + 1)
#             train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
#             # Must transfer to .cpu() tensor firstly for saving images
#             torchvision.utils.save_image(test_output.cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "recon_{}_{:.4g}dB.png".format(iterations + 1, test_psnr)))
#             print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(iterations + 1, max_iter, test_loss, test_psnr))

#     # Save final model
#     model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
#     torch.save({'net': model.state_dict(), \
#                 'enc': encoder.B, \
#                 'opt': optim.state_dict(), \
#                 }, model_name)