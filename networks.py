# import tensorflow as tf
import pickle
import numpy as np
# from tensorflow import keras
# from tensorflow.contrib.layers import batch_norm
# from networks import *
import torch
import torch.nn as nn

# Dense = keras.layers.Dense
# Activation = keras.layers.Activation
# UpSampling2D = keras.layers.UpSampling2D
# Flatten = keras.layers.Flatten
# Conv2D = keras.layers.Conv2D
# Conv2DTranspose = keras.layers.Conv2DTranspose
# Model = keras.models.Model
# LeakyReLU = keras.layers.LeakyReLU
# PReLU = keras.layers.PReLU
# add = keras.layers.add
torch.cuda.set_device(3)

def res_block_gen(model, kernal_size, filters, strides, istraining = False):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = batch_norm(model, is_training = istraining)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = batch_norm(model, is_training = istraining)
        
    model = add([gen, model])
    
    return model
    
def up_sampling_residual_block(model, kernel_size, filters, strides, istraining = False, scope = None):
    model = tf.layers.conv2d(model, filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")
    gen = model
    model = batch_norm(model, is_training = istraining)
    model = LeakyReLU(alpha = 0.2)(model)
    model = tf.layers.conv2d(model, filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")
    
    model = gen + model
    model = batch_norm(model, is_training = istraining)
    model = LeakyReLU(alpha = 0.2)(model)
    model = UpSampling2D(size = 2, interpolation = 'bilinear')(model)
    
    return model

    
def up_sampling_block(model, kernel_size, filters, strides, istraining = False, scope = None):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    
    model = tf.layers.conv2d(model, filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")
    model = batch_norm(model, is_training = istraining)
    model = LeakyReLU(alpha = 0.2)(model)
    model = UpSampling2D(size = 2, interpolation = 'bilinear')(model)
    
    return model




############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, params):
        if params['embedding'] == 'gauss':
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = self.B.cuda(3)
        else:
            raise NotImplementedError

    def embedding(self, x):
        cuda3 = torch.device('cuda:3')
        self.B = self.B.to(cuda3)
#         print(self.B.get_device(), "B device")
#         print(self.B.t().get_device(), "BT device")
#         print(x.get_device(), "x device")
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding



############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ Fourier Feature Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
#             self.linear.weight.data.fill_(0)
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        ############test whether adding a relu layer will boost performance#####################
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))
        ############test whether adding a relu layer will boost performance#####################
#         layers.append(nn.PReLU())
#         layers.append(nn.Dropout(p=0.95))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out

