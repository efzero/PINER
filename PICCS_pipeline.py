import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import yaml
import os
from prior_utils import *
from skimage.metrics import peak_signal_noise_ratio as psnr
num_projs = 25
# num_projs = 30
# num_projs = 20
output_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/models/'
recon_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/NeRP_adaptive/'
prior_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/cnn_recon_test/'
adaptive_prior_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/adaptive_prior/'
gt_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/gt_ood_cropped/'
# folder_name = '/data/bowen/SparseReconstruction/3d-ct-full-dose/proj_25_ood'
img_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/'

def set_state(config_file,state_name, value):
    doc = None
    with open(config_file, 'r') as f:
        doc = yaml.safe_load(f)
        doc[state_name] = value
        
    with open(config_file, 'w') as f:
        new_yaml = yaml.dump(doc,f)
        
    
    return 

prefix = 'c'
offset = 0
for i in range(1,300):

    # if i in [0,300]:
    #     continue
    if i >= 100:
        prefix = 'b'
        offset = -100
    if i >= 200:
        prefix = 'a'
        offset = -200
    
    print(i,"iteration")
    res_path = '/data/bowen/SparseReconstruction/3d-ct-full-dose/gt_ood_cropped/' + prefix + '_' + str(i+offset) + '.npy'
    recon_path = img_path + 'lasso-tv2/' + 'recon_' + prefix + '-' + str(i+offset) + 'noisy.npy'
    set_state('configs/ct_recon_tv.yaml', 'num_projs', num_projs)
    set_state('configs/ct_recon_tv.yaml', 'img_path', res_path)
    set_state('configs/ct_recon_tv.yaml', 'recon_name', recon_path)
    set_state('configs/ct_recon_tv.yaml', 'reg', 'lasso')
    # set_state('configs/ct_recon_tv.yaml', 'reg', 'res-lasso')
    set_state('configs/ct_recon_tv.yaml', 'prior_index', i)
    set_state('configs/ct_recon_tv.yaml', 'lr', 0.002)
    set_state('configs/ct_recon_tv.yaml', 'max_iter', 4000)
    
    os.system('python TV_prior_image.py --config configs/ct_recon_tv.yaml --output_path  /data/bowen/SparseReconstruction/3d-ct-full-dose/models')

    gt = np.load("/data/bowen/SparseReconstruction/3d-ct-full-dose/gt_ood_cropped/" + prefix + "_" + str(i+offset) + ".npy")
    img = np.load(recon_path)[0,:,:,0]
    print(psnr(gt, np.clip(crop_img(img),0,1)))
    # np.save(recon_path, np.clip(crop_img(img),0,1))
#     
