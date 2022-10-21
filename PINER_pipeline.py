import numpy as np
import torch
import numpy as np
import numpy.linalg as la
import yaml
import os
from prior_utils import *
from glob import glob
import subprocess
import ruptures as rpt

###screen 12416

output_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/models/'
output_path2 = '/data/bowen/SparseReconstruction/3d-ct-full-dose/model_twopass/'
recon_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/PINER_recon/'
prior_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test/'
prior_path2 = '/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test2/'
adaptive_prior_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/adaptive_prior/'
PINER_prior_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/PINERprior/'
gt_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/gt_ood_cropped/'
fbp_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/fbp_raw/'

num_projs = 25
prefix = 'c'
offset = 0



for img_ind in range(0,300):


    if img_ind >= 100: ####only for ground true image indexing
        prefix = 'b'
        offset = -100
    if img_ind >= 200:
        prefix = 'a'
        offset = -200
    gt_img_name = prefix + '_' + str(img_ind+offset) + '.npy'
    # set_state('configs/image_regression_adapt.yaml', 'lr', 5e-5)
    # set_state('configs/image_regression_adapt.yaml', 'img_path', fbp_path + 'test' + str(img_ind) + 'proj' + str(num_projs) + '.npy')
    # os.system('python train_image_regression_test.py --config configs/image_regression_adapt.yaml --output_path  /data/bowen/SparseReconstruction/3d-ct-full-dose/model_twopass')
    # os.system('python try_recon.py  --config configs/unet.yaml --output_path  /data/bowen/SparseReconstruction/3d-ct-full-dose/models')

    # input_ = np.load('adapts_prod.npy')
    # output_ = np.load('cnn_adapt_prod.npy')
    # gt_input = np.load(fbp_path + 'test' + str(img_ind) +  'proj' + str(num_projs) + '.npy')
    ###adapt_flag = True if using original input, False if using adapted input
    
    # adapt_flag, adapt_output = get_adaptation(input_, gt_input, output_, window_size = 7)
    # print(adapt_flag)
    # np.save(PINER_prior_path + 'adapted_input' + str(img_ind) + '_redo.npy', np.clip(crop_img(adapt_output),0,1))


    # set_state('configs/image_regression.yaml', 'img_path', PINER_prior_path + 'adapted_input' + str(img_ind) + '.npy')
    # set_state('configs/image_regression.yaml', 'num_projs', num_projs)
    # set_state('configs/image_regression.yaml', 'img_index', img_ind)
    # set_state('configs/image_regression.yaml', 'recon_path', adaptive_prior_path + 'img' + str(img_ind) + '.npy')
    # os.system('python train_image_regression.py --config configs/image_regression.yaml --output_path  /data/bowen/SparseReconstruction/3d-ct-full-dose/models')


    set_state('configs/ct_recon.yaml', 'img_path', gt_path + gt_img_name)
    set_state('configs/ct_recon.yaml', 'num_projs', num_projs)

    set_state('configs/ct_recon.yaml', 'recon_name', recon_path + "recon_" + str(img_ind) + ".npy")
    # set_state('configs/ct_recon.yaml', 'recon_name', recon_path + "recon_" + str(img_ind) + "physics.npy")
    set_state('configs/ct_recon.yaml', 'reg', 'None')
    set_state('configs/ct_recon.yaml', 'max_iter', 1000)
    set_state('configs/ct_recon.yaml', 'lr', 1e-5)
    set_state('configs/ct_recon.yaml', 'prior_index', img_ind)
    model_name = glob(output_path + 'phantomproj' + str(num_projs) + 'adaptedimg' + str(img_ind)  +   '/checkpoints/*')[0]
    # model_name = glob(output_path + 'phantomproj' + str(num_projs) + 'img' + str(img_ind)  +   '/checkpoints/*')[0]
    print(model_name)
    set_state('configs/ct_recon.yaml', 'pretrain_model_path', model_name)

    os.system('python train_ct_recon2.py --config configs/ct_recon.yaml --output_path  /data/bowen/SparseReconstruction/3d-ct-full-dose/models --pretrain')



    
    





