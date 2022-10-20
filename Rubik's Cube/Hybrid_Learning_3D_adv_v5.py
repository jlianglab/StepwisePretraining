#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from cmath import inf

"""
CUDA_VISIBLE_DEVICES=2 python -W ignore Hybrid_Learning_3D_exp.py \
--note 3drubik-32b-1 \
--arch Unet \
--decoder upsampling \
--backbone resnet18 \
--input_rows 64 \
--input_cols 64 \
--input_deps 32 \
--init random \
--nb_class 1 \
--verbose 1 \
--batch_size 16 \
--scale 32   \
--workers 4



python Hybrid_Learning_3D_exp_v2.py \
--note 3drubik-pre-transpose-only-recon-1 \
--arch Unet \
--decoder transpose \
--backbone resnet18 \
--input_rows 64 \
--input_cols 64 \
--input_deps 64 \
--init finetune \
--nb_class 1 \
--verbose 1 \
--batch_size 16 \
--scale 32 \
--weights "/home/zguo32/rubrics_genesis/Models/exp_rubic/Unet-resnet18-3drubik-32-3.h5" \
--r_lambda 10 \
--o_lambda 10 \
--discard


python Hybrid_Learning_3D_exp_v2.py \
--note 3drubik-recon-pre-transpose-recon-1 \
--arch Unet \
--decoder transpose \
--backbone resnet18 \
--input_rows 64 \
--input_cols 64 \
--input_deps 64 \
--init random \
--verbose 1 \
--batch_size 16 \
--scale 32 \
--weights "/home/zguo32/rubrics_genesis/Models/exp_rubic/Unet-resnet18-3drubik-recon-pre-transpose-recon-1.h5" \
--epoch 170


Train whole network with pre-trained encoder:
python -W ignore Hybrid_Learning_3D_exp_v4.py \
--note 3drubik-cross-loss-no-flip-then-recon-rog-11100 \
--arch Unet \
--decoder upsampling \
--backbone resnet18 \
--input_rows 64 \
--input_cols 64 \
--input_deps 64 \
--init finetune \
--nb_class 1 \
--verbose 1 \
--batch_size 16 \
--scale 32 \
--r_lambda 1 \
--o_lambda 1 \
--g_lambda 100 \
--e_lr 1e-5 \
--g_lr le-3 \
--recon \
--workers 64 \
--weights Models/exp_rubic/Unet-resnet18-3drubik-genesis-cross-loss-no-flip.h5


Train whole network with pre-trained encoder and Genesis Augmentation:
python -W ignore Hybrid_Learning_3D_exp_v4.py \
--note 3drubik-cross-loss-no-flip-then-recon-rog-11100 \
--arch Unet \
--decoder upsampling \
--backbone resnet18 \
--input_rows 64 \
--input_cols 64 \
--input_deps 64 \
--init finetune \
--nb_class 1 \
--verbose 1 \
--batch_size 16 \
--scale 32 \
--r_lambda 1 \
--o_lambda 1 \
--g_lambda 100 \
--recon \
--genesis \
--workers 64 \
--weights Models/exp_rubic/Unet-resnet18-3drubik-genesis-cross-loss-no-flip.h5

Train reconstruction network with pre-trained encoder:
python -W ignore Hybrid_Learning_3D_exp_v4.py \
--note 3drubik-cross-loss-no-flip-then-recon-rog-11100 \
--arch Unet \
--decoder upsampling \
--backbone resnet18 \
--input_rows 64 \
--input_cols 64 \
--input_deps 64 \
--init finetune \
--nb_class 1 \
--verbose 1 \
--batch_size 16 \
--scale 32 \
--r_lambda 1 \
--o_lambda 1 \
--g_lambda 100 \
--recon \
--discard \
--workers 64 \
--weights Models/exp_rubic/Unet-resnet18-3drubik-genesis-cross-loss-no-flip.h5




"""






# In[1]:

# 9/20/2020 write thread-safe generator
# 9/20/2020 convert to rubik genesis


import warnings
warnings.filterwarnings('ignore')
import os
import keras
import tensorflow as tf
print("Keras = {}".format(keras.__version__))
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tqdm import tqdm
import numpy as np

import sys
import math
from scipy.special import comb
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import threading

import copy
import shutil
from sklearn import metrics

import random
from random import shuffle
from utils.BilinearUpSampling import *
from ynet3d import *
from unet3d import *
from keras.callbacks import LambdaCallback, TensorBoard
from keras.callbacks import Callback
from glob import glob
from skimage.transform import resize
from optparse import OptionParser
from PIL import Image, ImageDraw, ImageFont

from functools import partial
from segmentation_models import Nestnet, Unet, Xnet
from model import *
from keras.utils import plot_model
from keras.utils import GeneratorEnqueuer
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, GlobalAveragePooling3D
from keras.layers.merge import _Merge
from keras import backend as K

# tensorboard log
from logger import Logger
import time
import datetime

from skimage.util.shape import view_as_blocks


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--arch", dest="arch", help="Unet", default=None, type="string")
parser.add_option("--init", dest="init", help="random | finetune | finetune_recon", default=None, type="string")
parser.add_option("--backbone", dest="backbone", help="the backbones", default='', type="string")
parser.add_option("--decoder", dest="decoder_block_type", help="transpose | upsampling", default="upsampling", type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
parser.add_option("--nb_class", dest="nb_class", help="number of class", default=1, type="int")
parser.add_option("--verbose", dest="verbose", help="verbose", default=0, type="int")
parser.add_option("--weights", dest="weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--gweights", dest="gweights", help="pre-trained generator weights", default=None, type="string")
parser.add_option("--dweights", dest="dweights", help="pre-trained discriminator weights", default=None, type="string")
parser.add_option('--resume_iters', dest="resume_iters", type=int, default=None, help='resume training from this step')
parser.add_option("--note", dest="note", help="notes of experiment setup", default="", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type="int")
parser.add_option("--scale", dest="scale", help="the scale of pre-trained data", default=32, type="int")
parser.add_option("--workers", dest="workers", help="number of workers", default=2, type="int")
parser.add_option("--epoch", dest="epoch", help="resume step", default=0, type="int")
parser.add_option("--e_lr", dest="e_lr", help="learning rate", default=1e-3, type="float")
#parser.add_option("--d_lr", dest="d_lr", help="learning rate", default=1e-3, type="float")
parser.add_option('--g_lr', dest="g_lr", type=float, default=1e-3, help='generator learning rate')
parser.add_option('--d_lr', dest="d_lr", type=float, default=1e-3, help='discriminator learning rate')
parser.add_option("--r_lambda", dest="r_lambda", help="loss weight for rotation", default=1.0, type="float")
parser.add_option("--o_lambda", dest="o_lambda", help="loss weight for order", default=1.0, type="float")
parser.add_option("--g_lambda", dest="g_lambda", help="loss weight for reconstruction", default=1.0, type="float")
parser.add_option("--discard", dest="discard", help="keep classification head?", action='store_true', default=False)
parser.add_option("--recon", dest="recon", help="add the reconstruction head?", action='store_true', default=False)
parser.add_option("--genesis", dest="genesis", help="train as genesis?", action='store_true', default=False)
parser.add_option('--loss', dest="loss", type="string", default=None, help='L1/L2 loss')

parser.add_option("--rescale_rate", dest="rescale_rate", help="chance to perform rescaling", default=0.0, type="float")
parser.add_option("--nonlinear_rate", dest="nonlinear_rate", help="chance to perform nonlinear", default=0.9, type="float")
parser.add_option("--denoise_rate", dest="denoise_rate", help="chance to perform denoise", default=0.0, type="float")
parser.add_option("--paint_rate", dest="paint_rate", help="chance to perform painting", default=0.9, type="float")
parser.add_option("--outpaint_rate", dest="outpaint_rate", help="chance to perform out-painting", default=0.8, type="float")
parser.add_option("--rotation_rate", dest="rotation_rate", help="chance to perform rotation", default=0.0, type="float")
parser.add_option("--flip_rate", dest="flip_rate", help="chance to perform flipping", default=0.9, type="float")
parser.add_option("--local_rate", dest="local_rate", help="chance to perform local shuffle pixel", default=0.0, type="float")

(options, args) = parser.parse_args()

assert options.backbone in ['',
                            'vgg16',
                            'vgg19',
                            'resnet18',
                            'resnet34',
                            'resnet50',
                            'resnet101',
                            'resnet152',
                            'resnext50',
                            'resnext101',
                            'densenet121',
                            'densenet169',
                            'densenet201',
                            'inceptionv3',
                            'inceptionresnetv2',
                            ]
assert options.arch in ['Unet',
                        'Nestnet',
                        'Xnet'
                       ]
assert options.init in ['random',
                        'finetune',
                        'finetune_recon',
                        'pre-trained',
                        'continue'
                       ]
assert options.decoder_block_type in ['transpose',
                                      'upsampling'
                                     ]

seed = 1
random.seed(seed)
model_path = "Models/exp_rubik++/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
print('weight  ---  ', options.weights)
print('iters  ---  ', options.resume_iters)
    
class setup_config():
    #DATA_DIR = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes"
    DATA_DIR = "/home/zguo32/DATA/Self_Learning_Cubes_64x64x64/"
    optimizer = "Adam"
    nb_epoch = 10000
    patience = 40
    lr = 1e-0
    train_fold=[0,1,2,3,4]
    valid_fold=[5,6]
    test_fold=[7,8,9]
    rotate_degree = [90, 180, 270]
    hu_max = 1000.0
    hu_min = -1000.0
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
                 rescale_rate=0.0,
                 nonlinear_rate=0.95,
                 denoise_rate=0,
                 paint_rate=0.6,
                 outpaint_rate=0.8,
                 inpaint_rate=0.2,
                 rotation_rate=0.0,
                 flip_rate=0.0,
                 local_rate=0.9,
                 verbose=1,
                 workers=2,
                 epoch=0,
                 e_lr=1e-3,
                 r_lambda=1,
                 o_lambda=1,
                 g_lambda=1,
                 loss='L1',
                 discard=False,
                 recon=False,
                 genesis=False,
                ):
        self.model = model
        self.backbone = backbone
        self.init = init
        self.exp_name = model + "-" + backbone + "-" + note
        self.data_augmentation = data_augmentation
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.decoder_block_type = decoder_block_type
        self.rescale_rate = rescale_rate
        self.nonlinear_rate = nonlinear_rate
        self.denoise_rate = denoise_rate
        self.paint_rate = paint_rate
        self.outpaint_rate = outpaint_rate
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.rotation_rate = rotation_rate
        self.flip_rate = flip_rate
        self.local_rate = local_rate
        self.nb_class = nb_class
        self.workers = workers
        self.epoch = epoch
        self.e_lr = e_lr
        self.r_lambda = r_lambda
        self.o_lambda = o_lambda
        self.g_lambda = g_lambda
        self.discard = discard
        self.recon = recon
        self.genesis = genesis
        self.loss = loss
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"
        # if self.init != "finetune":
            # self.weights = None
        # else:
            # self.weights = "imagenet"
        #if self.init != "finetune":
        #    self.

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

config = setup_config(model=options.arch,
                      backbone=options.backbone,
                      note=options.note,
                      decoder_block_type=options.decoder_block_type,
                      input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      batch_size=options.batch_size,
                      nb_class=options.nb_class,
                      verbose=options.verbose,
                      rescale_rate=options.rescale_rate,
                      # nonlinear_rate=options.nonlinear_rate,
                      denoise_rate=options.denoise_rate,
                      # paint_rate=options.paint_rate,
                      # outpaint_rate=options.outpaint_rate,
                      rotation_rate=options.rotation_rate,
                      # flip_rate=options.flip_rate,
                      # local_rate=options.local_rate,
                      workers=options.workers,
                      epoch=options.epoch,
                      e_lr=options.e_lr,
                      r_lambda=options.r_lambda,
                      o_lambda=options.o_lambda,
                      g_lambda=options.g_lambda,
                      discard=options.discard,
                      recon=options.recon,
                      nonlinear_rate = 0.9,
                      paint_rate = 0.9,
                      outpaint_rate = 0.8,
                      inpaint_rate = 0.2,
                      local_rate = 0.5,
                      flip_rate = 0.4,
                      genesis = options.genesis,
                      loss = options.loss,
                      
                     )
config.display()

exp_path = os.path.join(model_path, config.exp_name)
print(exp_path)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
    
sample_folder = os.path.join(model_path, "sample", config.exp_name)
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder)     

# In[2]:

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

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

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
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

    #reshape image from (8,32,32,32) to (64,64,64)
    
    def shape_trans(img):
        B, H, W, D = img.shape[0], img.shape[2], img.shape[3], img.shape[4]
        print(B,H,W,D)
        img = img.reshape(B,1,2*H,2*W,2*D)
        img_temp = view_as_blocks(img,(B,1,H,W,D))
        img_temp = img_temp.reshape(B,1,2*H,2*W,2*D)
        return img_temp

    diff = x!=y_recont
    same = x==y_recont
    A = y_recont[diff]
    A_Hat = fake_img[diff]
    B = y_recont[same]
    B_Hat = fake_img[same]
    
    mse_diff = mse(A,A_Hat)
    mse_same = mse(B,B_Hat)        
            
    fake_img = shape_trans(fake_img)
    y_recont = shape_trans(y_recont)
    x = shape_trans(x)
    
    blank_ver = np.ones((fake_img.shape[2],10))
    blank_hor = np.ones((10, fake_img.shape[2]*4+30))
    sample_img = []
    
    for i in range(3):
        sample_img.extend(blank_hor)
    
    for b in range(fake_img.shape[0]):
        for c in range(2):
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
    
def elastic_transform_batch(image):
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
    

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    # _, img_rows, img_cols, img_deps = x.shape
    img_rows, img_cols, img_deps = x.shape
    num_block = 100
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[noise_x:noise_x+block_noise_size_x, 
                            noise_y:noise_y+block_noise_size_y, 
                            noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[noise_x:noise_x+block_noise_size_x, 
                   noise_y:noise_y+block_noise_size_y, 
                   noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x    

def image_in_painting(x):
    img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2]) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    # cnt = 4
    # while cnt > 0 and random.random() < 0.95:
        # block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        # block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        # block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        # noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        # noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        # noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        # x[noise_x:noise_x+block_noise_size_x, 
          # noise_y:noise_y+block_noise_size_y, 
          # noise_z:noise_z+block_noise_size_z] = image_temp[noise_x:noise_x+block_noise_size_x, 
                                                           # noise_y:noise_y+block_noise_size_y, 
                                                           # noise_z:noise_z+block_noise_size_z]
        # cnt -= 1
    return x
    
    
class rubik_generator(keras.utils.Sequence):
    def __init__(self, data, permutation_set,batch_size,mode=None):
        self.L = len(data)
        self.H = data.shape[2]
        self.W = data.shape[3]
        self.D = data.shape[4]
        self.data = data
        self.permutation_set = permutation_set
        self.perm_len = len(permutation_set)
        self.batch_size = batch_size
        self.mode = mode
        self.lock = threading.Lock()
        self.idx = random.sample(range(self.L),self.L)
    def __iter__(self):
        return self
    def __len__(self):
        return int(self.L/self.batch_size)
    def on_epoch_end(self):
        self.idx = random.sample(range(self.L),self.L)     
    def __getitem__(self,index):
        with self.lock:
            RotAngel = [[0,0],[0,1],[0,2],[1,2]]
            batch_bag = self.data[self.idx[index*self.batch_size:(index+1)*self.batch_size]]
            batch_bag[:,:,27:37,:,:] = 0
            batch_bag[:,:,:,27:37,:] = 0
            batch_bag[:,:,:,:,27:37] = 0
            
            permutation_idx = random.sample(range(self.perm_len),self.batch_size)
            permutation_bag = self.permutation_set[permutation_idx]
            permutation_label = np.zeros([self.batch_size,self.perm_len])
            for idx,label in enumerate(permutation_idx):
                permutation_label[idx][label] = 1
            x = []
            y = []
            rot_bag = np.zeros([self.batch_size,8])
            for i,image in enumerate(batch_bag):
                # print(image.shape)
                #if random.random() < 0.33:
                #    temp = elastic_transform(image[0])
                #else:
                temp = image[0]
                temp = view_as_blocks(temp,(int(self.H/2),int(self.W/2),int(self.D/2)))
                # temp = view_as_blocks(image,(1,int(self.H/2),int(self.W/2),int(self.D/2)))
                temp = temp.reshape(8,int(self.H/2),int(self.W/2),int(self.D/2))
                temp_ori = copy.deepcopy(temp)
        
                for j,slice in enumerate(temp):
                    idx = np.random.randint(4)
                    if idx == 0:
                        pass
                    else: 
                        # Rubik Rotate
                        temp[j] = np.rot90(slice,k=2,axes=RotAngel[idx])
                        rot_bag[i][j] = 1
                        
                        #Elastic Transform
                        # temp[j] = elastic_transform(temp[j])
                        
                        # Flip
                        # temp[j], temp_ori[np.newaxis,j,:] = data_augmentation(temp[j], temp_ori[np.newaxis,j,:], config.flip_rate)
                        
                        if config.genesis:
                            
                            r = random.random()
                            
                            if r<=0.25:                                       
                                # Local Shuffle Pixel
                                temp[j] = local_pixel_shuffling(temp[j], 1)
                            elif 0.25<r<=0.5:
                                # Apply non-Linear transformation with an assigned probability
                                temp[j] = nonlinear_transformation(temp[j], 1)
                            elif 0.5<r<=0.75:
                                # Inpainting
                                temp[j] = image_in_painting(temp[j])
                            else:
                                # Outpainting
                                temp[j] = image_out_painting(temp[j])
                
                x.extend(temp[permutation_bag[i]].reshape(1,8,int(self.H/2),int(self.W/2),int(self.D/2)))
                y.extend(temp_ori.reshape(1,8,int(self.H/2),int(self.W/2),int(self.D/2)))
                
            # if config.save_samples is not None and status == "train" and random.random() < 0.01:
                # n_sample = random.choice( [i for i in range(config.batch_size)] )
                # sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
                # sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
                # sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
                # sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
                # final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
                # final_sample = final_sample * 255.0
                # final_sample = final_sample.astype(np.uint8)
                # file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
                # imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)    
        
            x = np.asarray(x)
            y = np.asarray(y)

            if self.mode=='discard':
                return (x,y)
            elif self.mode == 'recon':
                # print(x.shape, rot_bag.shape, permutation_bag.shape)
                return (x, [rot_bag, permutation_label, y])
            else:
                return (x, [rot_bag, permutation_label])
    
    
    
def rotate(matrix, degree):
    matrix = list(matrix)
    if abs(degree) not in [0, 90, 180, 270]:
        raise
    if degree == 0:
        return np.array(list(matrix))
    elif degree > 0:
        return rotate(zip(*matrix[::-1]), degree-90)
    else:
        return rotate(zip(*matrix)[::-1], degree+90)
        
def randrot(data,permutation_set,batch_size,mode=None):
    #input shape [B,1,H,W,D]
    #H must equal to W
    #print(data.shape)
    H, W, D = data.shape[2], data.shape[3], data.shape[4]
    print(H,W,D)
    # assert H != D,"H must equal to W"
    RotAngel = [[0,0],[0,1],[0,2],[1,2]]
    while True:
        # random sample images
        batch_bag = data[random.sample(range(data.shape[0]),batch_size)]        
                
        # x = [27,28,29,30,31,32,33,34,35,36,37]
        # y = [27,28,29,30,31,32,33,34,35,36,37]
        # z = [21,22,23,24,25,26,27,28,29,30,31]     
        
        # batch_bag[:,:,x,:,:] = 0
        # batch_bag[:,:,:,y,:] = 0
        # batch_bag[:,:,:,:,z] = 0
        
        batch_bag[:,:,27:37,:,:] = 0
        batch_bag[:,:,:,27:37,:] = 0
        batch_bag[:,:,:,:,27:37] = 0
        
        # random sample permutations
        permutation_idx = random.sample(range(permutation_set.shape[0]),batch_size)
        permutation_bag = permutation_set[permutation_idx]
        permutation_label = np.zeros([batch_size,1000])
        for idx,label in enumerate(permutation_idx):
            #print("permutation_idx %d",label)
            permutation_label[idx][label] = 1
        x = []
        y = []
        rot_bag = np.zeros([batch_size,8])
        for i,image in enumerate(batch_bag):
            temp = view_as_blocks(image,(1,int(H/2),int(W/2),int(D/2)))
            temp = temp.reshape(8,int(H/2),int(W/2),int(D/2))
            temp_ori = np.copy(temp)
            # y.extend(temp.reshape(1,8,int(H/2),int(W/2),int(D/2)))
            for j,slice in enumerate(temp):
                idx = np.random.randint(4)
                if idx == 0:
                    pass
                #elif idx == 1:
                    # able to rotate [90,180,270]
                    #temp[j] = np.rot90(slice,k=np.random.randint(3),axes=RotAngel[idx])
                else: 
                    # only able to rotate [180]
                    temp[j] = np.rot90(slice,k=2,axes=RotAngel[idx])
                    rot_bag[i][j] = 1
            
            #print(temp.shape)
            #print(rot_bag.shape)
            # Add the rotated blocks            
            x.extend(temp[permutation_bag[i]].reshape(1,8,int(H/2),int(W/2),int(D/2)))
            # y.extend(temp_ori[permutation_bag[i]].reshape(1,8,int(H/2),int(W/2),int(D/2)))
            y.extend(temp_ori.reshape(1,8,int(H/2),int(W/2),int(D/2)))
            
        x = np.asarray(x)
        y = np.asarray(y)
        # return shape [B,8,H/2,W/2,D/2]
        #print(x.shape)
        #print(rot_bag.shape)
        #print(permutation_label.shape)
        #y = permutation_bag
        #yield ([x,batch_bag], [rot_bag, permutation_label,permutation_bag])
        # yield (x, [rot_bag, permutation_label])
        # print(np.where(permutation_label==1))
        if mode=='discard':
            yield (x,y)
        elif mode == 'recon':
            yield (x, [rot_bag, permutation_label])
        else:
            yield (x, [rot_bag, permutation_label, y])
        


def generate_pair(img, batch_size):
    img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])

            # Rescale
            if random.random() < config.rescale_rate:
                pass
                crop_size_x, crop_size_y = random.randint(int(0.5*img_rows), int(0.9*img_rows)), random.randint(int(0.5*img_cols), int(0.9*img_cols))
                image_temp = copy.deepcopy(x[n])
                crop_x, crop_y = random.randint(0, img_rows-crop_size_x-1), random.randint(0, img_cols-crop_size_y-1)
                image_temp = image_temp[crop_x:crop_x+crop_size_x, crop_y:crop_y+crop_size_y, :]
                image_temp = resize(image_temp, 
                                    (img_rows, img_cols, img_deps), 
                                    preserve_range=True,
                                   )
                x[n], y[n] = image_temp, image_temp
            
            # Rotation
            if random.random() < config.rotation_rate:
                pass
                degree = random.choice(config.rotate_degree)
                x[n] = rotate(x[n], degree)
                y[n] = rotate(y[n], degree)
               
            # Flip
            cnt = 3
            while random.random() < config.flip_rate and cnt > 0:
                degree = random.choice([0, 1, 2])
                x[n] = np.flip(x[n], axis=degree)
                y[n] = np.flip(y[n], axis=degree)
                cnt = cnt - 1

            # Local Shuffle Pixel
            if random.random() < config.local_rate:
                pass
                image_temp = copy.deepcopy(x[n])
                orig_image = copy.deepcopy(x[n])
                num_block = 100
                for _ in range(num_block):
                    block_noise_size_x, block_noise_size_y, block_noise_size_z = random.randint(1, 6), random.randint(1, 6), random.randint(1, 6)
                    noise_x, noise_y, noise_z = random.randint(0, img_rows-block_noise_size_y), random.randint(0, img_cols-block_noise_size_x)
                    ind_list = [i for i in range(block_noise_size_x * block_noise_size_y)]
                    random.shuffle(ind_list)
                    for order, shuff in enumerate(ind_list):
                        image_temp[noise_x+order%block_noise_size_y, noise_y+order//block_noise_size_y, :] = orig_image[noise_x+shuff%block_noise_size_y, noise_y+shuff//block_noise_size_y, :]
                x[n] = image_temp
            
            # Non-Linear
            if random.random() < config.nonlinear_rate:
                points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
                xpoints = [p[0] for p in points]
                ypoints = [p[1] for p in points]
                xvals, yvals = bezier_curve(points, nTimes=100000)
                if random.random() < 0.5:
                    # Half change to get flip
                    xvals = np.sort(xvals)
                else:
                    xvals, yvals = np.sort(xvals), np.sort(yvals)
                x[n] = np.interp(y[n], xvals, yvals)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting for 50%
                    block_noise_size_x, block_noise_size_y, block_noise_size_z = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
                    noise_x, noise_y, noise_z = random.randint(3, img_rows-block_noise_size_x-3), random.randint(3, img_cols-block_noise_size_y-3), random.randint(3, img_deps-block_noise_size_z-3)
                    x[n, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, noise_z:noise_z+block_noise_size_z] = random.random()
                else:
                    # Outpainting for 50%
                    block_noise_size_x, block_noise_size_y, block_noise_size_z = img_rows - random.randint(10, 20), img_cols - random.randint(10, 20), img_deps - random.randint(10, 20)
                    noise_x, noise_y, noise_z = random.randint(3, img_rows-block_noise_size_x-3), random.randint(3, img_cols-block_noise_size_y-3), random.randint(3, img_deps-block_noise_size_z-3)
                    image_temp = copy.deepcopy(x[n])
                    x[n] = random.random()
                    x[n, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, noise_z:noise_z+block_noise_size_z]
                
        yield (x, y)


# learning rate schedule
# source: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch,part):
    if part=='Gen':
        if config.init == 'pre-trained':
            initial_lrate = 0.0001
        else:
            initial_lrate = options.g_lr
    else:
        initial_lrate = options.d_lr
    
    drop = 0.5
    epochs_drop = int(config.patience * 0.25)
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    print("current {} learning rate is {}".format(part,lrate))
    
    return lrate

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


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
    
    
umodel = unet_model_3d((1, config.input_rows/2, config.input_cols/2, config.input_deps/2), batch_normalization=True)
base_model = keras.models.Model(inputs = umodel.input, outputs = umodel.get_layer('depth_7_relu').output)
    
rubic_input = Input([8, config.input_rows/2, config.input_cols/2, config.input_deps/2])
split = Lambda( lambda x: tf.split(x,num_or_size_splits=8,axis=1))(rubic_input)
Siamese_out = []
recon_int = []
for i in range(8):
    recon_int.append(umodel(split[i]))
    Siamese_out.append(base_model(split[i]))
    
recon_out = Concatenate(axis=1,name='recon_out')(recon_int)    
net = Concatenate(axis=1)(Siamese_out)
net = GlobalAveragePooling3D(data_format='channels_first')(net)    
rot_out = Dense(8, activation='sigmoid',name='rot_out')(net)
order_out = Dense(1000, activation='softmax',name='order_out')(net)    
    
gen_base = keras.models.Model(inputs=rubic_input, outputs=[rot_out,order_out,recon_out])
#dis_base = unet_model_3d((2, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True)
dis_base = unet_model_3d((16, config.input_rows/2, config.input_cols/2, config.input_deps/2), batch_normalization=True)

if options.gweights is not None:
    print("Load the pre-trained generator weights from {}".format(options.gweights))
    gen_base.load_weights(options.gweights)
    #model_rubik = keras.models.load_model(options.gweights, custom_objects={"tf": tf})
    # if options.discard:
    # encoder weight
    #gen_base.layers[2].set_weights(model_rubik.layers[2].get_weights())
    # rot_out weight
    #gen_base.layers[5].set_weights(model_rubik.layers[5].get_weights())
    # order_out weight
    #gen_base.layers[6].set_weights(model_rubik.layers[6].get_weights())
    #gen_base.layers[7].set_weights(model_rubik.layers[7].get_weights())


if config.loss == 'L1':
    
    gen_base.compile(optimizer=keras.optimizers.SGD(lr=config.e_lr, momentum=0.9, decay=0.0, nesterov=False), 
              loss=["binary_crossentropy","categorical_crossentropy","MAE"],#loss=["MSE","MSE"],
              loss_weights=[config.r_lambda,config.o_lambda,config.g_lambda], 
              metrics = {'rot_out':'accuracy', 'order_out':'accuracy', 'recon_out':'MAE'})#["accuracy","MSE"])  
else:
    gen_base.compile(optimizer=keras.optimizers.SGD(lr=config.e_lr, momentum=0.9, decay=0.0, nesterov=False), 
              loss=["binary_crossentropy","categorical_crossentropy","MSE"],#loss=["MSE","MSE"],
              loss_weights=[config.r_lambda,config.o_lambda,config.g_lambda], 
              metrics = {'rot_out':'accuracy', 'order_out':'accuracy', 'recon_out':'MSE'})#["accuracy","MSE"])  


#building discriminator
base_out = dis_base.get_layer('depth_7_relu').output
x = Conv3D(1, 1)(base_out)
dis_model = keras.models.Model(inputs=dis_base.input, outputs=x)
#if options.dweights is not None:
#    print("Load discriminator weights from {}, continue training".format(options.dweights))
#    dis_model.load_weights(options.dweights)
#else:
#    print("Load rubik weight to discriminator from {}".format(options.weights))
#    model_rubik = keras.models.load_model(options.weights,custom_objects={'tf':tf})
#    dis_model.set_weights(model_rubik.layers[2].get_weights())
    
#real_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
#fake_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
#disarrange_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
real_inp = Input(shape=(8,config.input_rows/2,config.input_cols/2,config.input_deps/2))
fake_inp = Input(shape=(8,config.input_rows/2,config.input_cols/2,config.input_deps/2))
disarrange_inp = Input(shape=(8,config.input_rows/2,config.input_cols/2,config.input_deps/2))

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

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=20, decay_rate=0.9)
# disopt = keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
disopt = keras.optimizers.Nadam(lr=1e-4, beta_1=0.5, beta_2=0.999)

dis.compile(optimizer=disopt,
            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
            loss_weights=[1, 1, 10])


            
#building generator            
for layer in dis_model.layers:
    layer.trainable = False
dis_model.trainable = False

#disarrange_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
disarrange_inp = Input(shape=(8,config.input_rows/2,config.input_cols/2,config.input_deps/2))
gen_out = gen_base(disarrange_inp)

x = Concatenate(axis=1)([gen_out[2],disarrange_inp])           
real_fake = dis_model(x)

gen = keras.models.Model(disarrange_inp, [gen_out[0], gen_out[1], gen_out[2], real_fake])

# genopt = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                # beta_1=0.5,
                                # beta_2=0.999)
                                
genopt = keras.optimizers.Nadam(lr=1e-4, beta_1=0.5, beta_2=0.999)

                                                                
if config.loss == 'L1':
    gen.compile(optimizer = genopt,
            loss=["binary_crossentropy","categorical_crossentropy","MAE",wasserstein_loss],#loss=["MSE","MSE"],
            loss_weights=[config.r_lambda,config.o_lambda,config.g_lambda,1],
            metrics = {'rot_out':'accuracy', 'order_out':'accuracy', 'recon_out':'MAE'})
            
else:
     gen.compile(optimizer = genopt,
            loss=["binary_crossentropy","categorical_crossentropy","MSE",wasserstein_loss],#loss=["MSE","MSE"],
            loss_weights=[config.r_lambda,config.o_lambda,config.g_lambda,1],
            metrics = {'rot_out':'accuracy', 'order_out':'accuracy', 'recon_out':'MSE'})

for layer in dis_model.layers:
    layer.trainable = True            
dis_model.trainable = True

gen.summary()
dis.summary()    
    
gen.metrics_names
dis.metrics_names


x_train = []
for i,fold in enumerate(tqdm(config.train_fold+config.test_fold)):
    #s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x32_"+str(fold)+".npy"))
    s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x64_"+str(fold)+".npy"),mmap_mode='r')
    x_train.extend(s)
#x_train = np.asarray(x_train)
x_train = np.expand_dims(np.array(x_train), axis=1)
print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))

x_valid = []
for i,fold in enumerate(tqdm(config.valid_fold)):
    #s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x32_"+str(fold)+".npy"))
    s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x64_"+str(fold)+".npy"),mmap_mode='r')
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)
#x_valid = np.asarray(x_valid)
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

permutation_all = np.load("permutations_hamming_max_1000.npy")

logger = Logger(os.path.join(logs_path, config.exp_name))         
train_generator = rubik_generator(x_train, permutation_all,config.batch_size,mode='recon')
validation_generator = rubik_generator(x_valid, permutation_all,config.batch_size,mode='recon')

training_enq = tf.keras.utils.OrderedEnqueuer(train_generator,use_multiprocessing=False,shuffle=True)

#valid = -np.ones((config.batch_size, 1, 8, 8, 8))
#fake =  np.ones((config.batch_size, 1, 8, 8, 8))
#dummy = np.zeros((config.batch_size, 1, 8, 8, 8))
valid = -np.ones((config.batch_size, 1, 4, 4, 4))
fake =  np.ones((config.batch_size, 1, 4, 4, 4))
dummy = np.zeros((config.batch_size, 1, 4, 4, 4))

def restore_model(exp_path, config):
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(options.resume_iters))
    print(exp_path)
    G_path = os.path.join(exp_path, config.exp_name+'-{}-G.h5'.format(options.resume_iters))
    D_path = os.path.join(exp_path, config.exp_name+'-{}-D.h5'.format(options.resume_iters))
    gen_base.load_weights(G_path)
    dis_model.load_weights(D_path)
    
start_time = time.time()
total_iter = train_generator.__len__()  #// config.batch_size
#train_iter = iter(train_generator)
print(total_iter)
fake_idx = [0,15,31]

#best_mse = 1.0
best_loss = float('inf')

start_ep = 0
if options.resume_iters is not None:
    start_ep = options.resume_iters
    restore_model(exp_path, config)     
    
# training starts
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
    # yield (x, [rot_bag, permutation_label, y])
    
    
    for it in range(total_iter):
        x, [y_cls_rot, y_cls_order, y_recont] = next(train_gen)
        _, _, fake_img = gen_base.predict(x)
        
        
        if it == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            print('traing start now:{}'.format(et))
            #save_image(x,y_recont,fake_img,ep)
        
        dis_loss = dis.train_on_batch([y_recont, fake_img, x], [valid, fake, dummy])
        gen_loss = gen.train_on_batch(x, [y_cls_rot, y_cls_order, y_recont, valid])
        
        #print(len(dis_loss))
        #print(len(gen_loss))
            
        if it%100 == 0:
            loss = {}
            loss["dis_real"] = dis_loss[1]
            loss["dis_fake"] = dis_loss[2]
            loss["dis_gp"] = dis_loss[3]
            loss["gen_clc_rot"] = gen_loss[1]
            loss["gen_clc_order"] = gen_loss[2]
            #loss["gen_clc_rot"] = gen_loss["accuracy"]
            #loss["gen_clc_order"] = gen_loss["MSE"]
            #loss["gen_acc_clc_rot"] = gen_loss[5]
            #loss["gen_acc_clc_order"] = gen_loss[6]
            loss["gen_rec"] = gen_loss[3]
            loss["gen_fake"] = gen_loss[4]


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
    
    
    val_loss = gen_base.evaluate_generator(validation_generator, steps=int(validation_generator.__len__()), 
                                           max_queue_size=2*options.workers, workers=options.workers, verbose=1)
    print("Epoch {} validation MSE = {:.5f}".format(ep+1, val_loss[3]))
    
    #print(len(val_loss))
    v_loss = {}
    
    v_loss_total = config.r_lambda*val_loss[1] + config.o_lambda*val_loss[2] \
    + config.g_lambda*val_loss[3] + 1.0*val_loss[4]
    
    
    v_loss["validation_total"] = v_loss_total
    v_loss["validation_gen_clc_rot"] = val_loss[1]
    v_loss["validation_gen_clc_order"] = val_loss[2]
    #v_loss["validation_gen_acc_clc_rot"] = val_loss[5]
    #v_loss["validation_gen_acc_clc_order"] = val_loss[6]
    v_loss["validation_gen_rec"] = val_loss[3]
    v_loss["validation_gen_fake"] = val_loss[4]
    

    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log = "Epoch[{}], Elapsed [{}], Iteration [{}/{}]".format(ep+1, et, it, total_iter)
    for tag, value in v_loss.items():
        log += ", {}: {:.4f}".format(tag, value)
    print(log)
            
    for tag, value in v_loss.items():
        logger.scalar_summary(tag, value, it)
    
    #v_loss_total = config.r_lambda*val_loss[1] + config.o_lambda*val_loss[2] \
    #+ config.g_lambda*val_loss[3] + 1.0*val_loss[4]

    if best_loss > v_loss_total:
        dis_model.save_weights(os.path.join(exp_path,config.exp_name+'-{}-D.h5'.format(ep+1)))
        gen_base.save_weights(os.path.join(exp_path, config.exp_name+'-{}-G.h5'.format(ep+1)))
        print("validation {} improved from {}, saving model".format(v_loss_total,best_loss))
        best_loss = v_loss_total
    else:    
        print("validation {} did not improved from {}".format(v_loss_total,best_loss))




