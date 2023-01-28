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
from DnCNN import *



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
img_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/' 

# data_loader = get_data_loader("fbp_recons_test.npy", '/data/bowen/SparseReconstruction/3d-ct-full-dose/test_abdominal_imgs_ood_final.npy', batch_size = 16)

# data_loader = get_data_loader("nerp_recon_fbptest.npy", "nerp_recon_fbptest.npy",batch_size = 1)
# data_loader = get_data_loader("fbp_result_iter0.npy", "fbp_result_iter0.npy",batch_size = 1)

# data_loader = get_data_loader("test_smooth.npy", "test_smooth.npy", batch_size = 1)

# data_loader = get_data_loader(img_path + "combined_recon_iter1.npy", img_path + "combined_recon_iter1.npy", train = False, batch_size = 15)
# data_loader = get_data_loader(img_path+ 'fbp_imgs.npy', img_path+ 'fbp_imgs.npy', train = False, batch_size = 15)
# data_loader = get_data_loader("corrected_recon_iter1.npy", "corrected_recon_iter1.npy",batch_size = 1)
# data_loader = get_data_loader('rn_i0_nlx1.npy', 'rn_i0_nlx1.npy', batch_size = 1)

# data_loader = get_data_loader('cnntest.npy', 'cnntest.npy', batch_size = 1)
# data_loader = get_data_loader('fbp_test.npy', 'fbp_test.npy', train = False, batch_size = 15)
# data_loader = get_data_loader(img_path + 'fbp_cnn_recon.npy', img_path + 'fbp_cnn_recon.npy', train=False, batch_size=15)
# data_loader = get_data_loader(img_path + 'fbp_test_proj' + str(20) + '.npy', img_path + 'fbp_test_proj' + str(20) + '.npy', train = False, batch_size = 10)

# data_loader = get_data_loader('test_proj.npy', 'test_proj.npy', train = False, batch_size = 11) ###test adaptation to angles
# data_loader = get_data_loader('test_cnninput6.npy', 'test_cnninput6.npy', train = False, batch_size = 1)
# data_loader = get_data_loader('adapts_test.npy','adapts_test.npy', train = False, batch_size=10)
# data_loader = get_data_loader('adapts_test2.npy','adapts_test2.npy', train = False, batch_size=10)
# data_loader = get_data_loader('adapts_test3.npy','adapts_test3.npy', train = False, batch_size=10)
# data_loader = get_data_loader('adapts_test4.npy','adapts_test4.npy', train = False, batch_size=10)


# data_loader = get_data_loader("features_and_data/fbp_test.npy", "features_and_data/fbp_test.npy", train = False, batch_size=10)  #####for reconstruction with the original fbp test set

data_loader = get_data_loader('adapts_prod.npy','adapts_prod.npy', train = False, batch_size=10)  ####for production (input adaptation)

# data_loader = get_data_loader('cnn_adapt_prod.npy','cnn_adapt_prod.npy', train = False, batch_size=10)
# data_loader = get_data_loader('test_cnninput6_baseline.npy', 'test_cnninput6_baseline.npy', train = False, batch_size = 1)
# Load experiment setting

# data_loader = get_data_loader('features_and_data/fbp_test_complex_noise_unet.npy', 'features_and_data/fbp_test_complex_noise_unet.npy', train = False, batch_size = 10)  ####for black-box model complex-noise recon
# data_loader = get_data_loader('features_and_data/fbp_test_lowdose_unet.npy', 'features_and_data/fbp_test_lowdose_unet.npy', train = False, batch_size = 10)  ####for black-box model low-dose recon


opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']
model_name = "Unet1"
# model_name = "Unet_8192022_robust"
# model_name = "Unet_8212022_robust"
# model_name = "DnCNN_6262022"


cudnn.benchmark = True

output_folder = '/data/bowen/SparseReconstruction/3d-ct-full-dose/models'


train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
print(checkpoint_directory)


# # Setup model
model = Unet(use_dropout=True)
# model = Unet(use_dropout=True, num_down=3)
# model = DnCNN(channels = 1)
model_path = checkpoint_directory + '/model_000101.pt'
model.load_state_dict(torch.load(model_path)['net'])
model.eval()
model.cuda(2)


ret = np.zeros((300, 256, 256)) ###for black-box model reocn
# ret = np.zeros((100, 256,256)) 
# ret = np.zeros((1,256,256))  ###for single image recon
# ret = np.zeros((50,256,256)) ###for input adaptation
a,b = None, None

cur_head = 0
for it, (fbp_raw, gt) in enumerate(data_loader):
        
    fbp_raw = fbp_raw.transpose(1,3).float()
    gt = gt.transpose(1,3).float()
    fbp_raw = fbp_raw.cuda(2)
    gt = gt.cuda(2)
    test_data = (fbp_raw, gt)
    test_output = model(test_data[0])
    
    test_img = test_output.transpose(1,3).float()
    a = test_img.cpu().detach().numpy()
    b = gt.cpu().detach().numpy()
    
    ret[cur_head:cur_head+10, :,:] = a[:,:,:,0]
    cur_head += 10
    
# np.save(img_path + "cnn_projected_iter2.npy", ret)
# np.save(img_path + 'fbp_cnn_recon.npy', ret)
# np.save(img_path + 'fbp_cnn_recon2.npy', ret)
# np.save(img_path + 'fbp_cnn_recon_proj20.npy', ret)

# np.save("test_prior.npy", ret[0])
# np.save("test_adaptation6.npy", ret[0])
# np.save('test_adaptation6_baseline.npy', ret[0])

# np.save('test_fbp_gt_iter2.npy', b)

# np.save('rn_cnn_projected_iter2.npy', a)
# np.save('cnn_projected2.npy',a)
# np.save("cnn_adapt.npy", ret)
# np.save("cnn_adapt2.npy", ret)
# np.save("cnn_adapt3.npy", ret)
# np.save("cnn_adapt4.npy", ret)
np.save("cnn_adapt_prod.npy", ret) ###for input adaptation
# np.save("features_and_data/unet_recons_complex_noise_unet.npy", ret) ###for recon complex noise
# np.save("features_and_data/unet_recons_lowdose_unet.npy", ret) ###for recon low dose noise
# np.save("features_and_data/unet_recons_robust_continuous.npy", ret) ###for robust recon
# np.save("features_and_data/DnCNN_recon_lowdose.npy", ret) ###for recon lowdose dncnn


