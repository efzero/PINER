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

import numpy as np
from tqdm import tqdm
from prior_utils import *

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader, ct_parallel_project_2d_batch


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']
adapts =  np.zeros((50, 256, 256))

cudnn.benchmark = True

# Setup output folder
# output_folder = os.path.splitext(os.path.basename(opts.config))[0]
output_folder = '/data/bowen/SparseReconstruction/3d-ct-full-dose/models'
model_name = os.path.join(output_folder, config['data'] + 'proj' + str(config['num_projs']) + 'img' + str(config['img_index']))
# if not(config['encoder']['embedding'] == 'none'):
#     model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)
recon_path = config['recon_path']

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


# Setup input encoder:
encoder = Positional_Encoder(config['encoder'])

# Setup model
if config['model'] == 'SIREN':
    model = SIREN(config['net'])
elif config['model'] == 'FFN':
    model = FFN(config['net'])
else:
    raise NotImplementedError
model.cuda(3)
model.train()

# Setup optimizer
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


# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], -1, train=True, batch_size=config['batch_size'])
image_directory = '/home/bowen/Sparse_Reconstruction/img_regression'
for it, (grid, image) in enumerate(data_loader):
    
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda(3)  # [bs, h, w, 2], [0, 1]
    image = image.cuda(3)  # [bs, h, w, c], [0, 1]

    img_arr = image.detach().cpu().numpy()
    noise_level = noise_estimate(img_arr[0], pch_size=16)

    
    # Data loading 
    # Change training inputs for downsampling image
    test_data = (grid, image)
    train_data = (grid, image)
    input_xy = train_data[0]
    input_xy.requires_grad = True
    print(test_data[1].cpu().shape, train_data[1].cpu().shape)

    torchvision.utils.save_image(test_data[1].cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "test.png"))
    torchvision.utils.save_image(train_data[1].cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "train.png"))

    # Train model
    for iterations in range(max_iter):
        model.train()
        optim.zero_grad()

        train_embedding = encoder.embedding(train_data[0])  # [B, H, W, embedding*2]
        train_output = model(train_embedding)  # [B, H, W, 3]
        res_dx = train_output[:,:,1:,0] - train_output[:,:,:-1,0]
        res_dy = train_output[:,1:,:,0] - train_output[:,:-1,:,0]
        mse_loss = 0.5 * loss_fn(train_output, train_data[1])
        tv_loss = torch.mean(torch.abs(res_dx)) + torch.mean(torch.abs(res_dy))
        train_loss = mse_loss
        # train_loss = mse_loss + 1e-2*1*tv_loss
        if iterations % 20 == 0:
            # train_loss_num = mse_loss.detach().cpu().numpy()
            model.eval()
            with torch.no_grad():
                test_embedding = encoder.embedding(test_data[0])
                test_output = model(test_embedding)
                tv_loss_num = tv_loss.detach().cpu().numpy()
                train_psnr = -10 * torch.log10(2 * mse_loss).item()
                adapts[iterations//20] = test_output.detach().cpu().numpy()[0,:,:,0]
            # np.save('adapted_' + str(iterations//100) + '.npy', test_output.detach().cpu().numpy())

            # print(tv_loss_num, train_psnr)
            # if tv_loss_num > 0.085 and train_psnr > 18:
            #     np.save(recon_path, test_output.detach().cpu().numpy())
            #     break
        #     if train_loss_num < noise_level**2/2*1:
        #         np.save(recon_path, test_output.detach().cpu().numpy())
        #         break
#         res_dx = train_output
#         res_dx = train_output[:,:,1:,0] - train_output[:,:,:-1,0]
#         res_dy = train_output
#         res_dy[:,1:,:,0] = train_output[:,1:,:,0] - train_output[:,:-1,:,0]
        
#         image_gradient = grad(outputs= train_output, inputs= input_xy, grad_outputs = torch.ones(train_output.size()).cuda(3), create_graph = True)[0]
#         print(image_gradient.size())
#         train_loss = 0.5*(loss_fn(res_dx, train_data[1]))
        

        train_loss.backward()
        optim.step()

        # Compute training psnr
        if (iterations + 1) % config['log_iter'] == 0:
            train_psnr = -10 * torch.log10(2 * train_loss).item()
            train_loss = train_loss.item()

            train_writer.add_scalar('train_loss', train_loss, iterations + 1)
            train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss, train_psnr))

        # Compute testing psnr
        if (iterations + 1) % config['val_iter'] == 0:
            model.eval()
            with torch.no_grad():
                test_embedding = encoder.embedding(test_data[0])
                test_output = model(test_embedding)

                test_loss = 0.5 * loss_fn(test_output, test_data[1])
                test_psnr = - 10 * torch.log10(2 * test_loss).item()
                test_loss = test_loss.item()

            train_writer.add_scalar('test_loss', test_loss, iterations + 1)
            train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
            # Must transfer to .cpu() tensor firstly for saving images
            torchvision.utils.save_image(test_output.cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "recon_{}_{:.4g}dB.png".format(iterations + 1, test_psnr)))
            # np.save("recon_{}_{:.4g}dB.npy".format(iterations + 1, test_psnr), test_output.cpu().numpy())
            print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(iterations + 1, max_iter, test_loss, test_psnr))

        if iterations == max_iter-1:
            np.save(recon_path, test_output.detach().cpu().numpy())
            

        

    # Save final model
    model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
    torch.save({'net': model.state_dict(), \
                'enc': encoder.B, \
                'opt': optim.state_dict(), \
                }, model_name)

    

adapts[-1] = train_data[1].detach().cpu().numpy()[0,:,:,0]
# np.save('adapts_test2.npy', adapts)
# np.save('adapts_test3.npy', adapts)
# np.save('adapts_test4.npy', adapts)
np.save('adapts_prod.npy', adapts)



# grid, image = ds[0]
# grid = grid.unsqueeze(0).to(device)
# image = image.unsqueeze(0).to(device)

# downsample_ratio = 2
# test_data = (grid, image)  # [1, 512, 512, 2], [1, 512, 512, 3]
# train_data = (grid[:, ::downsample_ratio, ::downsample_ratio, :], image[:, ::downsample_ratio, ::downsample_ratio, :])  # [1, 256, 256, 2], [1, 256, 256, 3]
# torchvision.utils.save_image(train_data[1].cpu().permute(0, 3, 1, 2).data, f"outputs/phantom/train.jpeg")
# torchvision.utils.save_image(test_data[1].cpu().permute(0, 3, 1, 2).data, f"outputs/phantom/test.jpeg")

# # Downsample
# # exp = "downsample"
# # downsample_ratio = 2
# # train_data = (grid[:, ::downsample_ratio, ::downsample_ratio, :], image[:, ::downsample_ratio, ::downsample_ratio, :])  # [1, 256, 256, 2], [1, 256, 256, 3]

# # Randomsample
# exp = "randomsample"
# index_x = np.sort(np.random.permutation(np.arange(img_size))[:(img_size//2)])
# index_y = np.sort(np.random.permutation(np.arange(img_size))[:(img_size//2)])
# # train_data = (grid[:, :, index_x, :], image[:, :, index_x, :])
# # print(train_data[1].shape)
# # train_data = (grid[:, index_y, :, :], image[:, index_y, :, :])
# # print(train_data[1].shape)
# train_data = (grid[:, index_y, :, :][:, :, index_x, :], image[:, index_y, :, :][:, :, index_x, :])
# print(train_data[1].shape)
   
