# PINER: Prior-Informed Implicit Neural Representation Learning for Test-Time Adaptation in Sparse-View CT Reconstruction


## Contents

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Instructions for Running Code](#instructions-for-running-code)
- [License](#license)
- [Citation](#citation)

# 1. Overview

This repository provides the PyTorch code for our WACV 2023 papar [PINER: Prior-Informed Implicit Neural Representation Learning for Test-Time Adaptation in Sparse-View CT Reconstruction](https://openaccess.thecvf.com/content/WACV2023/html/Song_PINER_Prior-Informed_Implicit_Neural_Representation_Learning_for_Test-Time_Adaptation_in_WACV_2023_paper.html).

by [Bowen Song](https://web.stanford.edu/~bowens18/), Liyue Shen, Lei Xing

We propose a two-stage input-adaptation and output-correction framework with implicit neural representation learning for test-time adaptation for sparse-view CT reconstruction with unknown and varying noise levels. 

<p align="center">
  <img src="https://github.com/efzero/PINER/blob/master/networks/flowchart_revised_finalfinal_compressed.jpg" width="1200" height="400">
</p>

# 2. Installation Guide

Run 
Before running this package, users should have `Python`, `PyTorch`, and several python packages installed (`numpy`, `skimage`, `yaml`, `opencv`, `odl`) .


## Package Versions

This code functions with following dependency packages. The versions of software are, specifically:
```
python: 3.7.4
pytorch: 1.4.1
numpy: 1.19.4
skimage: 0.17.2
yaml: 0.1.7
opencv: 3.4.2
odl: 1.0.0.dev0
astra-toolbox
```


## Package Installment

Users should install all the required packages shown above prior to running the algorithm. Most packages can be installed by running following command in terminal on Linux. To install PyTorch, please refer to their official [website](https://pytorch.org). To install ODL, please refer to their official [website](https://github.com/odlgroup/odl).

```
pip install package-name
```



# 3. Instructions for Running Code


## 2D CT Reconstruction Experiment

The experiments of 2D CT image reconstruction use the 2D parallel-beam geometry.

### Step 1: Prior embedding

Represent 2D image by implicit network network. The prior image ([pancs_4dct_phase1.npz](./data/ct_data/pancs_4dct_phase1.npz): phase-1 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

```
python train_image_regression.py --config configs/image_regression.yaml
```

### Step 2: Network training

Reconstruct 2D CT image from sparsely sampled projections. The reconstruction target image ([pancs_4dct_phase6.npz](./data/ct_data/pancs_4dct_phase6.npz): phase-6 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

With prior embedding:
```
python train_ct_recon.py --config configs/ct_recon.yaml --pretrain
```

Without prior embedding:
```
python train_ct_recon.py --config configs/ct_recon.yaml
```

## 3D CT Reconstruction Experiment

The experiments of 3D CT image reconstruction use the 3D cone-beam geometry.

### Step 1: Prior embedding

Represent 3D image by implicit network network. The prior image ([pancs_4dct_phase1.npz](./data/ct_data/pancs_4dct_phase1.npz): phase-1 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

```
python train_image_regression_3d.py --config configs/image_regression_3d.yaml
```

### Step 2: Network training

Reconstruct 3D CT image from sparsely sampled projections. The reconstruction target image ([pancs_4dct_phase6.npz](./data/ct_data/pancs_4dct_phase6.npz): phase-6 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

With prior embedding:
```
train_ct_recon_3d.py --config configs/ct_recon_3d.yaml --pretrain
```

Without prior embedding:
```
python train_ct_recon_3d.py --config configs/ct_recon_3d.yaml
```

### Step 3: Image inference

Output and save the reconstruted 3D image after training is done at a specified iteration step.

With prior embedding:
```
python test_ct_recon_3d.py --config configs/ct_recon_3d.yaml --pretrain --iter 2000
```

Without prior embedding:
```
python test_ct_recon_3d.py --config configs/ct_recon_3d.yaml --iter 2000
```


# 5. Citation
If you find the code are useful, please consider citing the paper and the related paper.
```
@inproceedings{song2023piner,
  title={PINER: Prior-Informed Implicit Neural Representation Learning for Test-Time Adaptation in Sparse-View CT Reconstruction},
  author={Song, Bowen and Shen, Liyue and Xing, Lei},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1928--1938},
  year={2023}
}

@article{shen2022nerp,
  title={NeRP: implicit neural representation learning with prior embedding for sparsely sampled image reconstruction},
  author={Shen, Liyue and Pauly, John and Xing, Lei},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```
