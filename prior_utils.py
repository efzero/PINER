import numpy as np
from skimage import img_as_float
import time
import ruptures as rpt
import yaml
import numpy.linalg as la

def get_adaptation(input_, gt_img, output_, window_size = 5, tol = 5):
    dists = []
    change_point = -1
    in_distribution = True
    
    for img in input_:
        dists.append(la.norm(img - gt_img))
    inds = np.argsort(dists)[::-1]
    
    output_sorted = output_[inds]
    input_sorted = input_[inds]
    input_ch = input_sorted[window_size:] - input_sorted[:-window_size]
    output_ch = output_sorted[window_size:] - output_sorted[:-window_size]
    gradient = []
    for i in range(len(input_ch)):
        gradient.append(la.norm(output_ch[i]) / la.norm(input_ch[i]))
        
    np.save("features_and_data/gradient" +  str(window_size) + ".npy", np.array(gradient))
    gradient = np.array(gradient)
    gradient_std = gradient.std()
    bic = 2*gradient_std**2 * np.log(50) * (2)
    algo = rpt.Pelt(model="rbf").fit(np.array(gradient))
    result = algo.predict(pen=bic)
    adapted_pt_ind = np.argmin(gradient)
    

    prev = 0
    for ch_pt in result:
        if ch_pt > 0 and ch_pt < len(gradient)-1:
            change_point = ch_pt
            cur_section, prev_section = gradient[ch_pt:], gradient[prev:ch_pt]

            if cur_section.mean() > prev_section.mean(): ##increasing change point
                
                if len(cur_section) >= window_size and len(prev_section) >= window_size:
                    change_point = ch_pt
                    local_min = np.argmin(gradient[:change_point])
                    print(local_min, change_point, "local min change point")
                    return False, output_sorted[local_min], (cur_section.mean() - prev_section.mean())/gradient_std
            
            prev = change_point
            
                   

    print(change_point, adapted_pt_ind, len(gradient), "change point")
    ##########check in distribution#############
#     return True, output_sorted[-1]
    if change_point == -1:
        return True, output_sorted[-1], -1
    
    if change_point < adapted_pt_ind:
        return True, output_sorted[-1], -1
    
#     gradient_after_change = gradient[change_point:]
#     gradient_before_change = gradient[adapted_pt_ind:change_point]
    
#     if np.mean(gradient_before_change) > np.mean(gradient_after_change):
#         return True, output_sorted[-1]
    if np.mean(gradient[adapted_pt_ind:]) < np.mean(gradient[max(0, adapted_pt_ind - window_size):adapted_pt_ind]):
        return True, output_sorted[-1], -1
    return False, output_sorted[adapted_pt_ind],0 
    
    
#     ch_mean, ch_std = np.mean(gradient_after_change), (np.std(gradient_after_change) + np.std(gradient_before_change))/2
#     gradient_window = gradient[max(0, adapted_pt_ind - tol // 2): adapted_pt_ind + tol // 2]
#     adapted_mean = np.mean(gradient_window)
#     print(ch_mean, ch_std, adapted_mean, len(gradient_window), "adaptation")
#     if adapted_mean < ch_mean - ch_std*1.7:
#         return False, output_sorted[adapted_pt_ind]
#     else:
#         return True, output_sorted[-1]
    
def crop_img(img):
    gt = np.zeros((256,256))
    image = gt.copy()
    shape_min = min(image.shape)
    radius = shape_min // 2
    img_shape = np.array(image.shape)
    coords = np.array(np.ogrid[:gt.shape[0], :gt.shape[1]],
                            dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    return 0*outside_reconstruction_circle + img * (1-outside_reconstruction_circle)


def set_state(config_file,state_name, value):
    doc = None
    with open(config_file, 'r') as f:
        doc = yaml.safe_load(f)
        doc[state_name] = value
        
    with open(config_file, 'w') as f:
        new_yaml = yaml.dump(doc,f)
        
    
    return


def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)