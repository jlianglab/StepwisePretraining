"""
    Owner: Fatemeh Haghighi, Mohammad Reza Hosseinzadeh Taher
    Copyright  to ASU JLiang Lab
"""

#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

"""
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=1 python -W ignore train_proxy_ynet_mr.py --explanation mr --encoder_weights Unetimg_label_mr11.h5

"""


import warnings

warnings.filterwarnings('ignore')
import os
import keras

print("Keras = {}".format(keras.__version__))
import tensorflow as tf
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import pylab
import sys
import math
import SimpleITK as sitk
try:
    from scipy.misc import comb
except ImportError:
    from scipy.special import comb

from matplotlib import offsetbox
import matplotlib.pyplot as plt
# from photutils import BoundingBox
import copy
import shutil
from sklearn import metrics
from functools import partial
from PIL import Image, ImageDraw, ImageFont

import random
from sklearn.utils import shuffle
from ynet3d import *
from unet3d import *
from keras.callbacks import LambdaCallback, TensorBoard
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, GlobalAveragePooling3D
from glob import glob
from skimage.transform import resize
from optparse import OptionParser
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
# from segmentation_models import Nestnet, Unet, Xnet
from keras.utils import plot_model
# from datetime import datetime
import datetime
import time
from logger import Logger
import threading


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--arch", dest="arch", help="Ynet", default="Ynet", type="string")
parser.add_option("--init", dest="init", help="pre-trained/scratch", default="scratch", type="string")
parser.add_option("--decoder", dest="decoder_block_type", help="transpose | upsampling", default="upsampling",
                  type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type=int)
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type=int)
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type=int)
parser.add_option("--verbose", dest="verbose", help="verbose", default=1, type=int)
parser.add_option("--weights", dest="weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--unet_weights", dest="unet_weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--encoder_weights", dest="encoder_weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type=int)
parser.add_option("--run", dest="run", help="run number", default=1, type=int)
parser.add_option("--explanation", dest="explanation", help="explanation", default=None)
parser.add_option("--nb_classes", dest="nb_classes", help="nb_classes", default=44,type=int)
parser.add_option("--nb_patch_per_image", dest="nb_patch_per_image", type=int, default=1)
parser.add_option("--gweights", dest="gweights", help="pre-trained generator weights", default=None, type="string")
parser.add_option("--dweights", dest="dweights", help="pre-trained discriminator weights", default=None, type="string")
parser.add_option('--resume_iters', dest="resume_iters", type=int, default=None, help='resume training from this step')
parser.add_option('--g_lr', dest="g_lr", type=float, default=1e-3, help='generator learning rate')
parser.add_option('--d_lr', dest="d_lr", type=float, default=1e-3, help='discriminator learning rate')
parser.add_option('--workers', dest="workers", type=int, default=None, help='no. of workers')
parser.add_option('--abs_loss', dest="abs_loss", action='store_true', help='generator fake loss abs')


(options, args) = parser.parse_args()

assert options.arch in ['Unet',
                        'Nestnet',
                        'Xnet',
                        'Ynet',
                        'Adv-Ynet'
                        ]

assert options.decoder_block_type in ['transpose',
                                      'upsampling'
                                      ]

seed = 1
random.seed(seed)
model_path = "Models/proxy_Img_Img/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

def date_str():
	return datetime.now().__str__().replace("-", "_").replace(" ", "_").replace(":", "_")

class setup_config():
    optimizer = "Adam"
    nb_epoch = 10000
    patience = 20


    def __init__(self, model="Unet",
                 backbone="",
                 init="",
                 note="",
                 data_augmentation=True,
                 input_rows=64,
                 input_cols=64,
                 input_deps=32,
                 batch_size=64,
                 decoder_block_type=None,
                 nb_class=1,
                 verbose=1,
                 cls_classes=44,
                 run=1,
                 explanation="",
                 #  lr=0.001,
                 g_lr=0.001,
                 d_lr=0.001,
                 resume_iters=None,
                 ):
        self.init = init
        self.model = model
        self.backbone = backbone
        self.exp_name = "vw_ynet_"+explanation+str(run)
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.decoder_block_type = decoder_block_type
        self.nb_class = nb_class
        self.cls_classes=cls_classes
        self.g_lr=g_lr
        self.d_lr=d_lr
        self.resume_iters = resume_iters
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = setup_config(model=options.arch,
                      decoder_block_type=options.decoder_block_type,
                      input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      batch_size=options.batch_size,
                      verbose=options.verbose,
                      cls_classes=options.nb_classes,
                      run=options.run,
                      explanation=options.explanation,
                      resume_iters=options.resume_iters,
                      g_lr=options.g_lr,
                      d_lr=options.d_lr  
                      )
config.display()

sample_folder = os.path.join(model_path, "sample", config.exp_name)
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder) 
    
exp_path = os.path.join(model_path, config.exp_name)
print(exp_path)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

def elastic_transform(image):
    alpha = 991
    sigma = 8
    random_state = np.random.RandomState(None)
    shape_mrht = np.shape(image)

    dx = gaussian_filter((random_state.rand(*shape_mrht) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape_mrht) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape_mrht[0]), np.arange(shape_mrht[1]), np.arange(shape_mrht[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    transformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape_mrht)
    return transformed_image


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    # Suppose Input shape : (H,W,D)
    
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    
    if imageA.ndim == 3:
        H,W,D = imageA.shape
        err /= float(H * W * D)
    if imageA.ndim == 1:
        err /= float(len(imageA))    
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return err
    
def save_image(x,y_recont,fake_img,ep):
    diff = x!=y_recont
    same = x==y_recont
    A = y_recont[diff]
    A_Hat = fake_img[diff]
    B = y_recont[same]
    B_Hat = fake_img[same]
    
    mse_diff = mse(A,A_Hat)
    mse_same = mse(B,B_Hat)        
    
    blank_ver = np.ones((fake_img.shape[2],10))
    blank_hor = np.ones((10, fake_img.shape[2]*4+30))
    sample_img = []
    
    for i in range(3):
        sample_img.extend(blank_hor)
        
    for b in range(fake_img.shape[0]):
        for c in range(2):
                # temp_axis = [slice(None)]*fake_img.ndim
                # # for c in [0,1] ==> dim 4, [2,3]==> dim 3, [4,5] ==> dim 2, alternating from 0 and -1 ==> showing the
                # # images from all 6 faces
                # # temp_axis[4-c//2] = -1*(c%2)
                # # temp_axis[4-c//2] = -7*(c%2)+3 # alternating from -4 to 3
                # # temp_axis[0] = b
                # # temp_axis[1] = 0
                # temp_axis[4] = fake_idx[c] # alternating from -4 to 3
                # temp_axis[0] = b
                # temp_axis[1] = 0
              
                # gen_img = fake_img[tuple(temp_axis)]
                # ori_img = y_recont[tuple(temp_axis)]
                # disr_img = x[tuple(temp_axis)]
            
            gen_img = fake_img[b,0,:,:,c]
            ori_img = y_recont[b,0,:,:,c]
            disr_img = x[b,0,:,:,c]
            
            #difference map ==> normalize
            diff_img = gen_img - ori_img
            diff_img = (diff_img - np.min(diff_img))/(np.max(diff_img) - np.min(diff_img))                                   
            #suppress the low noises
                # low_cood = diff_img < 0.2
                # diff_img[low_cood]=0
                
                
            temp = np.concatenate((disr_img,blank_ver,gen_img,blank_ver,ori_img,blank_ver,diff_img),axis=1)
            sample_img.extend(temp)
            sample_img.extend(blank_hor)
            # gen_img_1 = fake_img[b,0,:,:,0]
            # ori_img_1 = y[b,0,:,:,0]
                # disr_img_1 = x[b,0,:,:,0]
    sample_img = np.expand_dims(np.array(sample_img),axis=2)
    sample_img = np.concatenate((sample_img,sample_img,sample_img),axis=2)
    sample_path = os.path.join(sample_folder,config.exp_name+'-{}-images.jpg'.format(ep+1))
    sample_img_save = Image.fromarray((sample_img * 255.).astype(np.uint8))
    font = ImageFont.truetype("arial.ttf", 8)
    d = ImageDraw.Draw(sample_img_save)
    d.text((5,0),"mse w. disarrange {:.4e}".format(mse_diff),fill=(0,0,0),font=font)
    d.text((150,0),"mse w.o disarrange {:.4e}".format(mse_same),fill=(0,0,0),font=font)
    d.text((10,15),"disr_img",fill=(0,0,0),font=font)
    d.text((85,15),"gen_img",fill=(0,0,0),font=font)
    d.text((160,15),"ori_img",fill=(0,0,0),font=font)
    d.text((235,15),"diff_img",fill=(0,0,0),font=font)
    
    
    sample_img_save.save(sample_path)
    print('Saved real and fake images into {}...'.format(sample_path))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, images_dir,infos_dir ,batch_size=16, dim=(64, 64, 32),nb_classes=44):
        self.images = np.load(images_dir)
        print(self.images.shape)
        self.infos=np.load(infos_dir)
        print(self.infos.shape)
        self.batch_size = batch_size
        self.dim = dim
        self.length = self.infos.shape[0]
        self.nb_classes=nb_classes
        self.lock = threading.Lock()

    def __len__(self):
        return int(np.floor(self.infos.shape[0]  / self.batch_size))

    def __getitem__(self, index):
        no_of_images_per_batch = int(self.batch_size)

        file_names = self.infos[index * no_of_images_per_batch:(index + 1) * no_of_images_per_batch,:]
        return self.luna_loader(file_names)

    def on_epoch_end(self):
        np.random.shuffle(self.infos)

    def luna_loader(self, file_list):
        nb_classes = self.nb_classes
        h=32
        w=32
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        input_depth = self.dim[2]
        x = np.zeros((self.batch_size, 1, input_rows, input_cols, input_depth), dtype="float")
        y_reconst = np.zeros((self.batch_size, 1, input_rows, input_cols, input_depth), dtype="float")
        y_cls = np.zeros((self.batch_size, nb_classes), dtype="int32")
        count = 0
        elastic_prob=1
        with self.lock:
            for i in range(file_list.shape[0]):
                patch = self.images[int(file_list[i,0]),:,:,:]

                label=int(file_list[i, 1])
                img_rows, img_cols, img_deps = patch.shape[0], patch.shape[1], patch.shape[2]
                y_reconst[count, :, :, :,:] =np.expand_dims(patch, axis=0)
                y_cls[count, :] = keras.utils.to_categorical(label, self.nb_classes)

                # Autoencoder
                cb = copy.deepcopy(patch)
                if random.random()<=elastic_prob:
                    patch_el = elastic_transform(cb)
                r=random.random()
                    # Local Shuffle Pixel
                if r<= 0.25:
                    image_temp = copy.deepcopy(patch_el)
                    orig_image = copy.deepcopy(patch_el)
                    num_block = 100
                    for _ in range(num_block):
                        block_noise_size_x = random.randint(1, img_rows // 10)
                        block_noise_size_y = random.randint(1, img_cols // 10)
                        block_noise_size_z = random.randint(1, img_deps // 10)
                        noise_x = random.randint(0, img_rows - block_noise_size_x)
                        noise_y = random.randint(0, img_cols - block_noise_size_y)
                        noise_z = random.randint(0, img_deps - block_noise_size_z)
                        window = orig_image[noise_x:noise_x + block_noise_size_x,
                                 noise_y:noise_y + block_noise_size_y,
                                 noise_z:noise_z + block_noise_size_z,
                                 ]
                        window = window.flatten()
                        np.random.shuffle(window)
                        window = window.reshape((block_noise_size_x,
                                                 block_noise_size_y,
                                                 block_noise_size_z))
                        image_temp[noise_x:noise_x + block_noise_size_x,
                                    noise_y:noise_y + block_noise_size_y,
                                    noise_z:noise_z + block_noise_size_z] = window
                        
                                                    
                                                    
                            
                        # ind_list = [i for i in range(block_noise_size_x * block_noise_size_y)]
                        # random.shuffle(ind_list)
                        # for order, shuff in enumerate(ind_list):
                        #     image_temp[noise_x + order % block_noise_size_y, noise_y + order // block_noise_size_y,
                        #     :] = orig_image[noise_x + shuff % block_noise_size_y, noise_y + shuff // block_noise_size_y,
                        #          :]
                    patch_el = image_temp

                    # Non-Linear
                elif 0.25 <r<=0.5:
                    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
                    xpoints = [p[0] for p in points]
                    ypoints = [p[1] for p in points]
                    xvals, yvals = bezier_curve(points, nTimes=100000)
                    if random.random() < 0.5:
                        # Half change to get flip
                        xvals = np.sort(xvals)
                    else:
                        xvals, yvals = np.sort(xvals), np.sort(yvals)
                    patch_el = np.interp(patch_el, xvals, yvals)

                    # Inpainting
                elif 0.5 < r <= 0.75:
                    block_noise_size_x, block_noise_size_y, block_noise_size_z = random.randint(10, 20), random.randint(
                        10, 20), random.randint(10, 20)
                    noise_x, noise_y, noise_z = random.randint(3, img_rows - block_noise_size_x - 3), random.randint(3,
                                                                                                                     img_cols - block_noise_size_y - 3), random.randint(
                        3, img_deps - block_noise_size_z - 3)
                    patch_el[ noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,
                    noise_z:noise_z + block_noise_size_z] = random.random()
                # & Outpainting
                else:
                    block_noise_size_x, block_noise_size_y, block_noise_size_z = img_rows - random.randint(10,
                                                                                                           20), img_cols - random.randint(
                        10, 20), img_deps - random.randint(10, 20)
                    noise_x, noise_y, noise_z = random.randint(3, img_rows - block_noise_size_x - 3), random.randint(3,
                                                                                                                     img_cols - block_noise_size_y - 3), random.randint(
                        3, img_deps - block_noise_size_z - 3)
                    image_temp = copy.deepcopy(patch_el)
                    patch_el[:,:,:] = random.random()
                    patch_el[ noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,
                    noise_z:noise_z + block_noise_size_z] = image_temp[ noise_x:noise_x + block_noise_size_x,
                                                            noise_y:noise_y + block_noise_size_y,
                                                            noise_z:noise_z + block_noise_size_z]


                x[count,:, :,:,:] = np.expand_dims(patch_el, axis=0)
                count += 1

            x, y_reconst,y_cls = shuffle(x,y_reconst, y_cls, random_state=0)
            
            return x, [y_reconst, y_cls]



# learning rate schedule
# source: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch,part):
    if part=='Gen':
        if config.init == 'pre-trained':
            initial_lrate = 0.0001
        else:
            initial_lrate = config.g_lr
    else:
        initial_lrate = config.d_lr
    drop = 0.5
    epochs_drop = int(config.patience * 0.8)
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    
    print("current {} learning rate is {}".format(part,lrate))

    return lrate
    
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def wasserstein_loss_abs(y_true, y_pred):
    return abs(K.mean(y_true * y_pred))

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1. - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)        
    
    
    
    

batch_size = int(config.batch_size)

# if config.model == "Ynet":
    # model = ynet_model_3d((1, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True,unet_weights=options.unet_weights,encoder_weights=options.encoder_weights,cls_classes=options.nb_classes)
# if options.weights is not None:
    # print("Load the pre-trained weights from {}".format(options.weights))
    # model.load_weights(options.weights)
# model.compile(optimizer=keras.optimizers.SGD(lr=config.lr, momentum=0.9, decay=0.0, nesterov=False),
              # loss={'reconst_output': 'mse', 'cls_output': 'categorical_crossentropy'},
              # loss_weights={'reconst_output': 100, 'cls_output': 1},
              # metrics={'reconst_output': ['mse', 'mae'], 'cls_output': ['categorical_crossentropy', 'accuracy']})

# model.compile(optimizer=keras.optimizers.Adam(lr=config.lr),
              # loss={'reconst_output': 'mse', 'cls_output': 'categorical_crossentropy'},
              # loss_weights={'reconst_output': 100, 'cls_output': 1},
              # metrics={'reconst_output': ['mse', 'mae'], 'cls_output': ['categorical_crossentropy', 'accuracy']})
              
              
              
def make_model():              
    if config.model == "Adv-Ynet":
        gen_base = ynet_model_3d((1, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True,unet_weights=options.unet_weights,encoder_weights=options.encoder_weights,cls_classes=options.nb_classes)          
        dis_base = unet_model_3d((2, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True)          
    if options.weights is not None:
        print("Load the pre-trained weights from {}".format(options.weights))
        gen_base.load_weights(options.weights)
        
    gen_base.compile(optimizer=keras.optimizers.Adam(lr=config.g_lr),
                  loss={'reconst_output': 'mae', 'cls_output': 'categorical_crossentropy'},
                  loss_weights={'reconst_output': 100, 'cls_output': 1},
                  metrics={'reconst_output': ['mse', 'mae'], 'cls_output': ['categorical_crossentropy', 'accuracy']})

    base_out = dis_base.get_layer('depth_7_relu').output
    x = Conv3D(1, 1)(base_out)
    dis_model = keras.models.Model(inputs=dis_base.input, outputs=x)
    if options.dweights is not None:
        print("Load the pre-trained discriminator weights from {}".format(options.dweights))
        dis_model.load_weights(options.dweights)              

    real_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
    fake_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
    disarrange_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))


    real_mix = Concatenate(axis=1)([real_inp,disarrange_inp])
    fake_mix = Concatenate(axis=1)([fake_inp,disarrange_inp])

    real_out = dis_model(real_mix)
    fake_out = dis_model(fake_mix)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_inp, fake_inp])
    interpolated_img = Concatenate(axis=1)([interpolated_img,interpolated_img])
    
    # Determine validity of weighted sample 
    validity_interpolated = dis_model(interpolated_img)
    
    # Use Python partial to provide loss function with additional
    # 'averaged_samples' argument
    partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names
    
    dis = keras.models.Model(inputs=[real_inp, fake_inp, disarrange_inp], outputs=[real_out, fake_out, validity_interpolated])
        
    dis.compile(optimizer=keras.optimizers.Adam(lr=config.d_lr),
                loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                # loss_weights=[1, 1, 10])
                loss_weights=[1, 1, 100])

    #building generator            
    #for layer in dis_model.layers:
    #    layer.trainable = False
    #dis_model.trainable = False
    
    disarrange_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
    
    gen_out = gen_base(disarrange_inp)
    
    
    #x = Concatenate(axis=1)([gen_out[0],disarrange_inp])           
    #real_fake = dis_model(x)
    
    #gen = keras.models.Model(disarrange_inp, [gen_out[0], gen_out[1], real_fake])
    gen = keras.models.Model(disarrange_inp, [gen_out[0], gen_out[1]])
    
    if config.init == 'pre-trained':
        gen.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                loss=["mae", 'categorical_crossentropy'],
                loss_weights=[100, 1])
                #loss=["mae", 'categorical_crossentropy', wasserstein_loss],
                # loss_weights=[1, 0.01])
                #loss_weights=[100, 1, 1])
            
    else:
        if options.abs_loss:
            gen.compile(optimizer=keras.optimizers.Adam(lr=config.g_lr),
                loss=["mae", 'categorical_crossentropy'],
                #loss=["mae", 'categorical_crossentropy', wasserstein_loss_abs],
                # loss_weights=[1, 0.01])
                loss_weights=[100, 1])
            print('abs_loss')
        else:    
            gen.compile(optimizer=keras.optimizers.Adam(lr=config.g_lr),
                loss=["mae", 'categorical_crossentropy'],
                #loss=["mae", 'categorical_crossentropy', wasserstein_loss],
                loss_weights=[100, 1])
                #loss_weights=[100, 1, 1])
            print('original_loss')
            
    for layer in dis_model.layers:
        layer.trainable = True            
    dis_model.trainable = True
    
    gen.summary()
    dis.summary()

    return gen, dis, dis_model, gen_base


            
gen,dis,dis_model,gen_base = make_model()    
    
# model.summary()

training_generator = DataGenerator("./data/sims_trains_multires.npy",'./data/trains_infos_multires.npy',
                                       batch_size=batch_size,dim=(config.input_rows,config.input_cols,config.input_deps),nb_classes=options.nb_classes)

# training_enq = tf.keras.utils.OrderedEnqueuer(training_generator)
training_enq = tf.keras.utils.OrderedEnqueuer(training_generator,use_multiprocessing=False,shuffle=True)


                                       
total_iter = training_generator.length  // batch_size
print(total_iter)

validation_generator = DataGenerator("./data/sims_validation_multires.npy",'./data/validation_infos_multires.npy',
                                         batch_size=batch_size,dim=(config.input_rows,config.input_cols,config.input_deps),nb_classes=options.nb_classes)

# valid_enq = tf.keras.utils.OrderedEnqueuer(validation_generator)
# valid_enq = tf.keras.utils.OrderedEnqueuer(validation_generator,use_multiprocessing=False,shuffle=True)
                             
                                         
                                         

# valid = -np.ones((config.batch_size, 1, 16, 8, 8))
# fake =  np.ones((config.batch_size, 1, 16, 8, 8))
# dummy = np.zeros((config.batch_size, 1, 16, 8, 8)) # Dummy gt for gradient penalty                            


valid = -np.ones((config.batch_size, 1, 8, 8, 4))
fake =  np.ones((config.batch_size, 1, 8, 8, 4))
dummy = np.zeros((config.batch_size, 1, 8, 8, 4))


def restore_model(exp_path, config):
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(config.resume_iters))
    print(exp_path)
    G_path = os.path.join(exp_path, config.exp_name+'-{}-G.h5'.format(config.resume_iters))
    D_path = os.path.join(exp_path, config.exp_name+'-{}-D.h5'.format(config.resume_iters))
    gen_base.load_weights(G_path)
    dis_model.load_weights(D_path)
    
    
start_time = time.time()
total_iter = training_generator.length  // batch_size
train_iter = iter(training_generator)

print(total_iter)

best_loss = 400.0

start_ep = 0
if config.resume_iters is not None:
    start_ep = config.resume_iters
    restore_model(exp_path, config)
    
logger = Logger(os.path.join(logs_path, config.exp_name))
    
fake_idx = [0,15,31]
    
for ep in range(start_ep, config.nb_epoch):
    
    # del gen,dis,dis_model,gen_base
    # K.clear_session()
    # tf.compat.v1.reset_default_graph()
    
    
    # gen,dis,dis_model,gen_base = make_model()
    n_disc_trainable = len(dis.trainable_weights)
    n_gen_trainable = len(gen_base.trainable_weights)
    
    print("Epoch", ep+1, "/", config.nb_epoch)
    print("discriminator trainable weights before epoch", ep+1, ":", n_disc_trainable)
    print("generator trainable weights before epoch", ep+1, ":", n_gen_trainable)
    # val_loss = gen_base.evaluate_generator(validation_generator, steps=int(validation_generator.length//config.batch_size), verbose=1)
    # print("Epoch {} validation MSE = {:.5f}".format(ep+1, val_loss[2]))
    
    # val_loss = gen_base.evaluate_generator(valid_gen, steps=int(validation_generator.length //config.batch_size), verbose=1)
    # print("Epoch {} validation MSE = {:.5f}".format(ep+1, val_loss[2]))
    
    
    curr_dis_lr = step_decay(ep,'Dis')
    curr_gen_lr = step_decay(ep,'Gen')
    K.set_value(dis.optimizer.lr, curr_dis_lr)        
    K.set_value(gen.optimizer.lr, curr_gen_lr) 
    
    training_enq.start(workers = options.workers,max_queue_size=2*options.workers)
    train_gen = training_enq.get()
    # valid_enq.start(workers = options.workers,max_queue_size=2*options.workers)
    # valid_gen = valid_enq.get()      
    
    for it in range(total_iter):
        x, [y_recont, y_cls] = next(train_gen)
        fake_img, _ = gen_base.predict(x)
       
        # save_image(x,y_recont,fake_img,ep)
        
        dis_loss = dis.train_on_batch([y_recont, fake_img, x], [valid, fake, dummy])
        gen_loss = gen.train_on_batch(x, [y_recont, y_cls, valid])

    
        if it%10 == 0:
            loss = {}
            loss["dis_real"] = dis_loss[1]
            loss["dis_fake"] = dis_loss[2]
            loss["dis_gp"] = dis_loss[3]
            loss["gen_rec"] = gen_loss[1]
            loss["gen_clc"] = gen_loss[2]
            loss["gen_fake"] = gen_loss[3]

            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Epoch[{}], Elapsed [{}], Iteration [{}/{}]".format(ep+1, et, it, total_iter)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)
            
            for tag, value in loss.items():
                logger.scalar_summary(tag, value, it)
    
    print("discriminator trainable weights before epoch", ep+1, ":", n_disc_trainable)
    print("discriminator trainable weights after epoch", ep+1, ":", len(dis.trainable_weights))
    print("generator trainable weights before epoch", ep+1, ":", n_gen_trainable) 
    print("generator trainable weights after epoch", ep+1, ":", len(gen.trainable_weights))
    
    n_disc_trainable = len(dis.trainable_weights)
    n_gen_trainable = len(gen_base.trainable_weights)
                
    
    if (ep+1) % 2 == 0:
        print(fake_img.shape)
        save_image(x,y_recont,fake_img,ep)
    
    
    val_loss = gen_base.evaluate_generator(validation_generator, steps=int(validation_generator.length //config.batch_size), 
                                           max_queue_size=2*options.workers, workers=options.workers, verbose=1)
    print("Epoch {} validation MSE = {:.5f}".format(ep+1, val_loss[2]))
    
    v_loss = {}
    
    v_loss["validation_gen_rec"] = val_loss[1]
    v_loss["validation_gen_clc"] = val_loss[2]
    v_loss["validation_gen_fake"] = val_loss[3]

    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log = "Epoch[{}], Elapsed [{}], Iteration [{}/{}]".format(ep+1, et, it, total_iter)
    for tag, value in v_loss.items():
        log += ", {}: {:.4f}".format(tag, value)
    print(log)
            
    for tag, value in v_loss.items():
        logger.scalar_summary(tag, value, it)
    
    v_loss_total = 100.0*val_loss[2] + 1.0*val_loss[1] + 1.0*val_loss[3]

    if best_loss > v_loss_total:
        dis_model.save_weights(os.path.join(exp_path,config.exp_name+'-{}-D.h5'.format(ep+1)))
        gen_base.save_weights(os.path.join(exp_path, config.exp_name+'-{}-G.h5'.format(ep+1)))
        best_loss = v_loss_total
        
    # K.clear_session()


    # train_gen.join()
    # valid_gen.join()
                                         
                                         
# plot_model(model, to_file=os.path.join(model_path, config.exp_name+".png"))
# if os.path.exists(os.path.join(model_path, config.exp_name + ".txt")):
    # os.remove(os.path.join(model_path, config.exp_name + ".txt"))
# with open(os.path.join(model_path, config.exp_name + ".txt"), 'w') as fh:
    # model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

# shutil.rmtree(os.path.join(logs_path, config.exp_name), ignore_errors=True)
# if not os.path.exists(os.path.join(logs_path, config.exp_name)):
    # os.makedirs(os.path.join(logs_path, config.exp_name))

# tbCallBack = TensorBoard(log_dir=os.path.join(logs_path, config.exp_name),
                         # histogram_freq=0,
                         # write_graph=True,
                         # write_images=True,
                        # )
# tbCallBack.set_model(model)
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               # patience=config.patience,
                                               # verbose=0,
                                               # mode='min',
                                               # )
# check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, config.exp_name + ".h5"),
                                              # monitor='val_loss',
                                              # verbose=1,
                                              # save_best_only=True,
                                              # mode='min',
                                              # )
# lrate = keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
# callbacks = [check_point, early_stopping,tbCallBack, lrate]

# In[ ]:

# while config.batch_size > 1:
#     # To find a largest batch size that can be fit into GPU
#     try:
# model.fit_generator(generator=training_generator,
                            # validation_data=validation_generator,
                            # steps_per_epoch=training_generator.length  // batch_size,
                            # validation_steps=validation_generator.length  // batch_size,
                            # epochs=config.nb_epoch,
                            # max_queue_size=20,
                            # workers=7,
                            # use_multiprocessing=True,
                            # shuffle=False,
                            # verbose=config.verbose,
                            # callbacks=callbacks,
                            # )
    #     break
    # except tf.errors.ResourceExhaustedError as e:
    #     config.batch_size = int(config.batch_size / 2.0)
    #     print("\n> Batch size = {}".format(config.batch_size))

# max_queue_size=20, workers=4, use_multiprocessing=True, shuffle=True,
