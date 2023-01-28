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

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader, ct_parallel_project_2d_batch
torch.cuda.set_device(3)


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
reg = config['reg']
PINER_prior_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/PINERprior/'
ind = int(config['prior_index'])
############testing noise###############

# noise = np.load("/data/bowen/SparseReconstruction/3d-ct-full-dose/noise.npy")[ind:(ind+1),:,:,0] #1,25,256,1
# noise = np.load('features_and_data/simulated_noise.npy') ####gaussian noise for unet
# noise = np.load('features_and_data/simulated_noise_large.npy') ####gaussian noise for dncnn

noise = np.load('features_and_data/lowdose_noise_unet.npy')
noise = noise.swapaxes(1,2)
noise = noise[ind:(ind+1), :,:] #1, 25, 256
print(np.mean(noise**2), noise.shape)
noise = noise*256
########################################



#############load prior image####################
# prior = np.load('/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test/cnn_recon_test' + str(ind) + '.npy')
# prior = np.load('/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test2/cnn_recon_test' + str(ind) + '.npy')
# prior = np.load('/data/bowen/SparseReconstruction/3d-ct-full-dose/adaptive_prior/img'+str(ind) + 'npy.npy')

# prior = np.load(PINER_prior_path + 'adapted_input' + str(ind) + '.npy')   ##unet adp
# prior = np.load(PINER_prior_path + 'adapted_input' + str(ind) + 'noisy_redo.npy')


# prior = np.load(PINER_prior_path + 'adapted_output' + str(ind) + '_lowdosenoise_unet.npy') ###unet low dose
# prior = np.load(PINER_prior_path + 'adapted_output' + str(ind) + '_lowdosenoise_dncnn.npy') ###unet low dose
prior = np.load(PINER_prior_path + 'adapted_input' + str(ind) + '_robust_continuous.npy')
prior = prior.reshape((1,256,256,1))

# prior = np.load("/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon/prior" + str(ind) + ".npy")
################################################################################



cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
if opts.pretrain: 
    output_subfolder = config['data'] + '_pretrain'
else:
    output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder)
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

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

# Load pretrain model
if opts.pretrain:
    model_path = config['pretrain_model_path']
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['net'])
    encoder.B = state_dict['enc']
    print('Load pretrain model: {}'.format(model_path))

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
    NotImplementedError


    
# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], -1, train=True, batch_size=config['batch_size'])

proj_noise = torch.from_numpy(noise).to(dtype = torch.float32).cuda(3)
prior = torch.from_numpy(prior).to(dtype = torch.float32).cuda(3)
angles = np.linspace(0, np.pi, config['num_projs'], endpoint=False)
radon = Radon(256, angles, clip_to_circle=True)


for it, (grid, image) in enumerate(data_loader):
    print(type(grid), type(image))
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda(3)  # [bs, h, w, 2], [0, 1]
    image = image.cuda(3)  # [bs, h, w, c], [0, 1]
    
    
    print(grid.shape, image.shape)

    projs = radon.forward(image[:,:,:,0])
    sinogram = (projs + proj_noise).detach().cpu().numpy()[0]/256
    noise_level = noise_estimate(sinogram)
    if noise_level < 1.5e-3:
        noise_level *= .9
    print(noise_level)


    test_data = (grid, image)
    train_data = (grid, projs)
    # Train model
    for iterations in range(max_iter):
        model.train()
        optim.zero_grad()
        input_xy = train_data[0]
        train_embedding = encoder.embedding(input_xy)  # [B, H, W, embedding*2]
        train_output = model(train_embedding)  # [B, H, W, 3]
        train_projs = radon.forward(train_output[:,:,:,0])

        train_loss = 0
# #         ###########################TEST SPARSITY CONSTRAINT METHOD#############################################
        if reg == 'lasso': 
        
            recovered = train_output + prior
            res_dx = recovered[:,:,1:,0] - recovered[:,:,:-1,0]
            res_dy = recovered[:,1:,:,0] - recovered[:,:-1,:,0]
#             print('using lasso')

####1.5, 3,,,,0.75, 1.5
#             train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1]) +5e-5*256**2*3*torch.mean(torch.abs(train_output))
            train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1]) +5e-5*256**2*1.5*torch.mean(torch.abs(train_output)) + \
    5e-5*256**2*3*(torch.mean(torch.abs(res_dx)) + torch.mean(torch.abs(res_dy)))
#             train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1]) +5e-5*256**2*1.5*(torch.mean(torch.abs(res_dx)) + torch.mean(torch.abs(res_dy)))
    
    
        elif reg == 'nerpdirect':
            res = train_output - prior
            res_dx = res[:,:,1:,0] - res[:,:,:-1,0]
            res_dy = res[:,1:,:,0] - res[:,:-1,:,0]
            
            train_loss = 0.5*loss_fn(train_projs + proj_noise, train_data[1])
#             + 5e-5*256**2*3*(torch.mean(torch.abs(res_dx)) + torch.mean(torch.abs(res_dy)))
            
            
        elif reg == 'lasso_ata':
            train_loss = 0.5*loss_fn(radon.backprojection(train_projs+proj_noise), radon.backprojection(train_data[1])) + 5e-5*torch.mean(torch.abs(train_output)**2)
            
        elif reg == 'ridge_ata':
            train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1]) + 5e-5*0*torch.mean(train_output**2)
            
        elif reg == 'priorcnn':
            train_loss = 0.5*loss_fn(train_projs + proj_noise, train_data[1]) + 5e-5*256**2*3*torch.mean(torch.abs(prior - train_output))

        elif reg == 'ablation':
            mse_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1])/256/256
            train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1])/256

        else:
            recovered = train_output
            res_dx = recovered[:,:,1:,0] - recovered[:,:,:-1,0]
            res_dy = recovered[:,1:,:,0] - recovered[:,:-1,:,0]
            
            mse_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1])/256/256
            train_loss = 0.5*loss_fn(train_projs+proj_noise, train_data[1])/256 + 5e-5*3*256*torch.mean(torch.abs(train_output - prior)) 
#             + 5e-5*256*1.5*(torch.mean(torch.abs(res_dx)) + torch.mean(torch.abs(res_dy)))

###hyperparameters not only depends on noise level but also on some other stuff
        # train_loss.backward()
        # optim.step()

        # Compute training psnr
        if iterations == 0 or (iterations + 1) % config['log_iter'] == 0:
            train_psnr = -10 * torch.log10(2 * mse_loss).item()
            # train_loss2 = train_loss.item()
            train_loss2 = mse_loss.item()
            # if train_loss2 < 4.5e-6*(.80)*1/9:
            if train_loss2 < noise_level**2/2*.9:
            # if train_loss2 < noise_level**2/2*1:
            # if train_loss2 < noise_level**2/2*.85:
                with torch.no_grad():
                    test_embedding = encoder.embedding(test_data[0])
                    test_output = model(test_embedding)
                    np.save(recon_name, test_output.cpu().numpy())
                    break
            

            train_writer.add_scalar('train_loss', train_loss2, iterations + 1)
            train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss2, train_psnr))

        # Compute testing psnr
        if iterations == 0 or (iterations + 1) % config['val_iter'] == 0 or iterations == (max_iter-1):
            model.eval()
            with torch.no_grad():
                test_embedding = encoder.embedding(test_data[0])
                test_output = model(test_embedding)
                
                print(np.max(test_output.cpu().numpy()), "peak signal value")
                np.save(recon_name, test_output.cpu().numpy())
                np.save("ground_truth_residual.npy", test_data[1].cpu().numpy())

#                 test_loss = 0.5 * loss_fn(test_output, test_data[1])######previously
                test_loss = 0.5*torch.nn.MSELoss()(test_output, test_data[1])
                test_psnr = - 10 * torch.log10(2 * test_loss).item()
                test_loss = test_loss.item()

            train_writer.add_scalar('test_loss', test_loss, iterations + 1)
            train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
            # Must transfer to .cpu() tensor firstly for saving images
            torchvision.utils.save_image(test_output.cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "recon_{}_{:.4g}dB.png".format(iterations + 1, test_psnr)))
            print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(iterations + 1, max_iter, test_loss, test_psnr))
        
        train_loss.backward()
        optim.step()
    
    # Save final model
    model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
    torch.save({'net': model.state_dict(), \
                'enc': encoder.B, \
                'opt': optim.state_dict(), \
                }, model_name)
    
torch.cuda.empty_cache()