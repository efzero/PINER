import os
import yaml
import math
import numpy as np
# import tensorflow as tf

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from data import ImageDataset, ImageDataset_2D, ImageDataset_3D


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory



def get_data_loader(data, img_path, img_dim, img_slice,
                    train, batch_size, 
                    num_workers=4, 
                    return_data_idx=False):
    
    if data == 'phantom':
        dataset = ImageDataset(img_path, img_dim)
    elif '3d' in data:
        dataset = ImageDataset_3D(img_path, img_dim)  # load the whole volume
    else:
        dataset = ImageDataset_2D(img_path, img_dim, img_slice)
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=train, 
                        drop_last=train, 
                        num_workers=num_workers)
    return loader


def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor = tensor[0, slice_idx, ...].permute(0, 3, 1, 2).cpu().data  # [c, 1, h, w]
    image_grid = vutils.make_grid(tensor, nrow=image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)



def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (B, H, W, C)
    coordinates: (2, ...)
    '''
    # h = input.shape[0]
    # w = input.shape[1]
    bs, h, w, c = input.size()

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)

    f00 = input[:, co_floor[0], co_floor[1], :]
    f10 = input[:, co_floor[0], co_ceil[1], :]
    f01 = input[:, co_ceil[0], co_floor[1], :]
    f11 = input[:, co_ceil[0], co_ceil[1], :]
    d1 = d1[None, :, :, None].expand(bs, -1, -1, c)
    d2 = d2[None, :, :, None].expand(bs, -1, -1, c)

    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    
    return fx1 + d2 * (fx2 - fx1)



def map_coordinates_tf(input, coordinates):
    ''' tensorflow version of scipy.ndimage.interpolation.map_coordinates
    input: (B, H, W, C)
    coordinates: (2, ...)
    '''
    # h = input.shape[0]
    # w = input.shape[1]
    bs, h, w, c = 1,256,256,1

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates
    co_floor = np.floor(coordinates).astype(int)
    co_ceil =np.ceil(coordinates).astype(int)
    d1 = (coordinates[1] - co_floor[1].astype(float))
    d2 = (coordinates[0] - co_floor[0].astype(float))
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)
    co_floor = tf.constant(co_floor, dtype = tf.int32)
    co_ceil = tf.constant(co_ceil, dtype = tf.int32)

    
#     f00 = input[:, co_floor[0], co_floor[1], :]
    input_minus = input[0,:,:,0]
    f00 = tf.gather_nd(input_minus, tf.stack((co_floor[0], co_floor[1]), -1))
    f10 = tf.gather_nd(input_minus, tf.stack((co_floor[0], co_ceil[1]), -1))
    f01 = tf.gather_nd(input_minus, tf.stack((co_ceil[0], co_floor[1]), -1))
    f11 = tf.gather_nd(input_minus, tf.stack((co_ceil[0], co_ceil[1]), -1))
    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    
    return tf.expand_dims(tf.expand_dims(fx1 + d2 * (fx2 - fx1), 0), -1)


def ct_parallel_project_2d_tf(img, theta):
    bs, h, w, c = 1,256,256,1
    x, y = np.meshgrid(np.arange(h) / h - 0.5, np.arange(w) / w - 0.5)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    # Reverse back to index [0, w]
    x_rot = (x_rot + 0.5) * w
    y_rot = (y_rot + 0.5) * h
    sample_coords = np.stack([y_rot, x_rot], axis=0)  # [2, h, w] y is 0, x is 1
    # Map the input array to new coordinates by interpolation: spline interpolation order. Boundary extension mode='constant'.
    img_resampled = map_coordinates_tf(img, sample_coords)
    proj = tf.reduce_mean(img_resampled, axis=1)
    return proj


def ct_parallel_project_2d_batch_tf(img, thetas):
    '''
    img: input tensor [B, H, W, C]
    thetas: list of projection angles
    '''
    projs = []
    for theta in thetas:
    	proj = ct_parallel_project_2d_tf(img, theta)
    	projs.append(proj)
    projs = tf.concat(projs, axis=1)  # [b, num, w, c]

    return projs
    


def ct_parallel_project_2d(img, theta):
    bs, h, w, c = img.size()

    # (y, x)=(i, j): [0, w] -> [-0.5, 0.5]
    device = torch.device("cuda:3")
    torch.cuda.set_device(device)
    y, x = torch.meshgrid([torch.arange(h, dtype=torch.float32) / h - 0.5, torch.arange(w, dtype=torch.float32) / w - 0.5])
    y = y.cuda()
    x = x.cuda()

    # Rotation transform matrix: simulate parallel projection rays
    # After rotate xy-axis for theta angle, what is the new (x', y') cooridinates transfromed from original (x, y) cooridnates?
    x_rot = x * torch.cos(theta) - y * torch.sin(theta)
    y_rot = x * torch.sin(theta) + y * torch.cos(theta)

    # Reverse back to index [0, w]
    x_rot = (x_rot + 0.5) * w
    y_rot = (y_rot + 0.5) * h

    # Resampled (x, y) index of the pixel on the projection ray-theta
    sample_coords = torch.stack([y_rot, x_rot], dim=0).cuda()  # [2, h, w]
    # Map the input array to new coordinates by interpolation: spline interpolation order. Boundary extension mode='constant'.
    img_resampled = map_coordinates(img, sample_coords) # [b, h, w, c]

    # Compute integral projections along rays
    proj = torch.mean(img_resampled, dim=1, keepdim=True) # [b, 1, w, c]
    proj.to(device)

    return proj


def ct_parallel_project_2d_batch(img, thetas):
    '''
    img: input tensor [B, H, W, C]
    thetas: list of projection angles
    '''
    projs = []
    for theta in thetas:
    	proj = ct_parallel_project_2d(img, theta)
    	projs.append(proj)
    projs = torch.cat(projs, dim=1)  # [b, num, w, c]

    return projs


def random_sample_gaussian_mask(shape, sample_ratio):
    '''Get binary mask for randomly sampling k-space with normal distribution.
    shape: image or k-space shape (2D or 3D) (N, N, N)
    sample_ratio: downsampling ratio 
    '''
    # Mean and Cov for normal distribtuion centered at the the center of the spectrum
    mean = np.array(shape) // 2  # mean at the center
    cov = np.eye(len(shape)) * (2 * shape[0])  # variance 

    # Sample coordinates (x, y, z) independent to each other with mean = 256 and var = 1024 within range [0, 512]
    nsamp = int(sample_ratio * np.prod(shape))
    samps = np.random.multivariate_normal(mean, cov, size=nsamp).astype(np.int32)  # [N, 3]
    # Within shape index range
    samps = np.clip(samps, 0, shape[0]-1)

    mask = np.zeros(shape)
    inds = []
    for i in range(samps.shape[-1]):
        inds.append(samps[..., i])
    
    mask[tuple(inds)] = 1.
    print('Downsample sparsity:', np.sum(np.abs(mask)) / np.prod(mask.shape))

    return torch.tensor(mask).astype(torch.complex64)


def random_sample_uniform_mask(shape, sample_ratio):
    '''Get binary mask for randomly sampling k-space with normal distribution.
    shape: tuple of image shape (2D or 3D) (N, N, N)
    sample_ratio: downsampling ratio 
    '''
    random_array = np.random.rand(*shape)  # uniform distribution over [0, 1)
    mask = random_array < sample_ratio
    mask = mask.astype(np.float32)

    print('Acceleration factor:', 1.0 / sample_ratio)
    print('Downsample sparsity:', np.sum(mask) / np.prod(mask.shape))

    return torch.tensor(mask)


def mri_fourier_transform_2d(image, mask=None):
    '''
    image: input tensor [B, H, W, C]
    mask: mask tensor [H, W]
    '''
    spectrum = torch.fft.fftn(image, dim=(1, 2))

    # K-space spectrum has been shifted to shift the zero-frequency component to the center of the spectrum
    spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))

    # # Downsample k-space
    # spectrum = spectrum * mask[None, :, :, None]

    return spectrum


def mri_fourier_transform_3d(image, mask=None):
    '''
    image: input tensor [B, C, H, W, 1]
    mask: mask tensor [C, H, W]
    '''
    spectrum = torch.fft.fftn(image, dim=(1, 2, 3))

    # K-space spectrum has been shifted to shift the zero-frequency component to the center of the spectrum
    spectrum = torch.fft.fftshift(spectrum, dim=(1, 2, 3))

    # # Downsample k-space
    # spectrum = spectrum * mask[None, :, :, None]

    return spectrum

def complex2real(spectrum):
    real_array = torch.log(torch.abs(spectrum) ** 2)

    return real_array


"""Github code: https://github.com/sifeluga/PDvdi/blob/master/PDvdiSampler.py """
class PoissonSampler2D:
    # those are never changed
    f2PI = math.pi * 2.
    # minimal radius (packing constant) for a fully sampled k-space
    fMinDist = 0.634

    def __init__(self, M=128, N=128, AF=12.0, fPow=2.0, NN=47, aspect=0, tInc=0.):
        self.M = M  # Pattern width
        self.N = N  # Pattern height
        self.fAF = AF  # Acceleration factor (AF) > 1.0
        self.fAspect = aspect
        if aspect == 0:
            self.fAspect = M / N  # Ratio of weight and height (non-square image)
        self.NN = NN  # Number of neighbors (NN) ~[20;80]
        self.fPow = fPow  # Power to raise of distance from center ~[1.5;2.5]
        self.tempInc = tInc  # Temporal incoherence [0;1]

        self.M2 = round(M / 2)  # Center of width
        self.N2 = round(N / 2)  # Center of height
        # need to store density matrix
        self.density = np.zeros((M, N), dtype=np.float32)
        self.targetPoints = round(M * N / AF)

        # init varDens
        if (self.fPow > 0):
            self.variableDensity()

    def variableDensity(self):
        """Precomputes a density matrix, which is used to scale the location-dependent
        radius used for generating new samples.
         """
        fNorm = 1.2 * math.sqrt(pow(self.M2, 2.) + pow(self.N2 * self.fAspect, 2.))

        # computes the euclidean distance for each potential sample location to the center
        for j in range(-self.N2, self.N2, 1):
            for i in range(-self.M2, self.M2, 1):
                self.density[i + self.M2, j + self.N2] = (
                1. - math.sqrt(math.pow(j * self.fAspect, 2.) + math.pow(i, 2.)) / fNorm)

        # avoid diving by zeros
        self.density[(self.density < 0.001)] = 0.001
        # raise scaled distance to the specified power (usually quadratic)
        self.density = np.power(self.density, self.fPow)
        accuDensity = math.floor(np.sum(self.density))

        # linearly adjust accumulated density to match desired number of samples
        if accuDensity != self.targetPoints:
            scale = self.targetPoints / accuDensity
            scale *= 1.0
            self.density *= scale
            self.density[(self.density < 0.001)] = 0.001
            # plt.pcolormesh(self.density)
            # plt.colorbar
            # plt.show()

    def addPoint(self, ptN, fDens, iReg):
        """Inserts a point in the sampling mask if that point is not yet sampled
        and suffices a location-depdent distance (variable density) to
        neighboring points. Returns the index > -1 on success."""
        ptNew = np.around(ptN).astype(np.int, copy=False)
        idx = ptNew[0] + ptNew[1] * self.M

        # point already taken
        if self.mask[ptNew[0], ptNew[1]]:
            return -1

        # check for points in close neighborhood
        for j in range(max(0, ptNew[1] - iReg), min(ptNew[1] + iReg, self.N), 1):
            for i in range(max(0, ptNew[0] - iReg), min(ptNew[0] + iReg, self.M), 1):
                if self.mask[i, j] == True:
                    pt = self.pointArr[self.idx2arr[i + j * self.M]]
                    if pow(pt[0] - ptN[0], 2.) + pow(pt[1] - ptN[1], 2.) < fDens:
                        return -1

        # success if no point was too close
        return idx

    def generate(self, seed, accu_mask=None):

        # set seed for deterministic results
        np.random.seed(seed)

        # preset storage variables
        self.idx2arr = np.zeros((self.M * self.N), dtype=np.int32)
        self.idx2arr.fill(-1)
        self.mask = np.zeros((self.M, self.N), dtype=bool)
        self.mask.fill(False)
        self.pointArr = np.zeros((self.M * self.N, 2), dtype=np.float32)
        activeList = []

        # inits
        count = 0
        pt = np.array([self.M2, self.N2], dtype=np.float32)

        # random jitter of inital point
        jitter = 4
        pt += np.random.uniform(-jitter / 2, jitter / 2, 2)

        # update: point matrix, mask, current idx, idx2matrix and activeList
        self.pointArr[count] = pt
        ptR = np.around(pt).astype(np.int, copy=False)
        idx = ptR[0] + ptR[1] * self.M
        self.mask[ptR[0], ptR[1]] = True
        self.idx2arr[idx] = count
        activeList.append(idx)
        count += 1

        # uniform density
        if (self.fPow == 0):
            self.fMinDist *= self.fAF

        # now sample points
        while (activeList):
            idxp = activeList.pop()
            curPt = self.pointArr[self.idx2arr[idxp]]
            curPtR = np.around(curPt).astype(np.int, copy=False)

            fCurDens = self.fMinDist
            if (self.fPow > 0):
                fCurDens /= self.density[curPtR[0], curPtR[1]]

            region = int(round(fCurDens))

            # if count >= self.targetPoints:
            #    break

            # try to generate NN points around an arbitrary existing point
            for i in range(0, self.NN):
                # random radius and angle
                fRad = np.random.uniform(fCurDens, fCurDens * 2.)
                fAng = np.random.uniform(0., self.f2PI)

                # generate new position
                ptNew = np.array([curPt[0], curPt[1]], dtype=np.float32)
                ptNew[0] += fRad * math.cos(fAng)
                ptNew[1] += fRad * math.sin(fAng)
                ptNewR = np.around(ptNew).astype(np.int, copy=False)
                # continue when old and new positions are the same after rounding
                if ptNewR[0] == curPtR[0] and ptNewR[1] == curPtR[1]:
                    continue

                if (ptNewR[0] >= 0 and ptNewR[1] >= 0 and ptNewR[0] < self.M and ptNewR[1] < self.N):
                    newCurDens = self.fMinDist / self.density[ptNewR[0], ptNewR[1]]
                    if self.fPow == 0:
                        newCurDens = self.fMinDist
                    if self.tempInc > 0 and accu_mask is not None:
                        if accu_mask[ptNewR[0], ptNewR[1]] > self.density[
                            ptNewR[0], ptNewR[1]] * seed + 1.01 - self.tempInc:
                            continue
                    idx = self.addPoint(ptNew, newCurDens, region)
                    if idx >= 0:
                        self.mask[ptNewR[0], ptNewR[1]] = True
                        self.pointArr[count] = ptNew
                        self.idx2arr[idx] = count
                        activeList.append(idx)
                        count += 1

        print("Generating finished with " + str(count) + " points.")

        return torch.tensor(self.mask.astype(np.float32)).to(torch.complex64)


# def write_samples(str,mask):
#     f = open(str,'w')
#     for i in range(0,mask.shape[0]):
#         for j in range(0, mask.shape[1]):
#             if mask[i,j] > 0:
#                 f.write('%d %d\n' % (i, j))
#     f.close()


def coordsto1d(x,y):
    return x*256 + y

def rotation_matrix(degree):
    return np.array([[np.cos(degree), -np.sin(degree)], [np.sin(degree), np.cos(degree)]])

def get_rotated_coords(coords_diag, degree):
    rot_mat = rotation_matrix(degree)
    new_coords = rot_mat@(coords_diag-127.5)+127.5
    existing_coords = set()
    new_valid_coords = []
    
    for coord in new_coords.T: ###2d coords
        x,y = coord
        x,y = round(x), round(y)
        if x >= 0 and x <= 255 and y >= 0 and y <= 255: #valid coord
            coord_ind = coordsto1d(x,y)
            #check duplicates
            if coord_ind not in existing_coords:
                new_valid_coords.append(coord_ind)
                existing_coords.add(coord_ind)
    return new_valid_coords
            
        
def get_Radon_transform_A(num_projs):
    A = np.zeros((1,num_projs*256,256*256))
    degree_increment = np.pi/num_projs
    
    thetas = np.linspace(0, np.pi, num_projs+1)[:-1]
    degree_cur = 0
    for i in range(num_projs):
        for j in range(256):
            coords_diag = np.array([np.ones(358)*j, np.arange(0-51,256+51)])
            degree_cur = thetas[i]
#             print(degree_cur, "degree_cur")
            indices = get_rotated_coords(coords_diag, degree_cur)
            A[0, i*256+j, indices] = 1/len(indices)
#         degree_cur += degree_increment
        
    return A