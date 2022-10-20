#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

"""
CUDA_VISIBLE_DEVICES=3 python -W ignore Hybrid_Learning_3D_exp.py \
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
--workers 8


python Jigsaw_v1_clean.py \
--note Jigsaw_encoder_decoder_fully_connect_l2_1 \
--arch Unet --decoder upsampling  --input_rows 64 \
--input_cols 64 --input_deps 64 --init pre-trained \
--nb_class 1 --verbose 1 --batch_size 16 --scale 32 \
--workers 32 --clc_head fully_connect --loss L1 \
--exp_part en_decoder \
--weights Models/exp_rubik++/encoder/Unet-resnet18-Jigsaw_encoder_fully_connect.h5





"""

# In[1]:

# 4/13/2020 Change log:
#(1) compute MSE at different places ==> changed / unchanged //Check
#(2) change batch shape from (BZ,1,128,64,64) to (BZ,2,64,64,64)  //Check
#(3) compare L2+adv and L2, L1+adv and L1
#(4) add difference map output/ground_truth  //Check

# 4/29/2020 Change log:
#(5) use pre-trained encoder (Rubik Cube)

# 5/31/2020 change log:
#(6) unet with transpose

# 6/13/2020 change log:
# (7) change adam learning rate to [g_lr = 0.0001, Beta1 = 0.5, Beta2 = 0.999] / keras example = [g_lr = 0.001, Beta1 = 0.9, Beta2 = 0.999]
# (8) change loss_weights gen from [200, 1] to [1, 0.1];






import warnings
warnings.filterwarnings('ignore')
import os
# from tensorflow import keras
import keras
import tensorflow as tf
print("Keras = {}".format(keras.__version__))
# import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import copy

import pylab
import sys
import math
import SimpleITK as sitk
from scipy.special import comb

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import copy
import shutil
from sklearn import metrics
import threading

import random
from random import shuffle
from utils.BilinearUpSampling import *
from unet3d import *
from keras.callbacks import LambdaCallback, TensorBoard
from glob import glob
from skimage.transform import resize
from optparse import OptionParser
from PIL import Image, ImageDraw, ImageFont

from functools import partial
from segmentation_models import Nestnet, Unet, Xnet
from model import *
from keras.utils import plot_model
from keras.utils import GeneratorEnqueuer
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, GlobalAveragePooling3D,Conv3D,Activation,BatchNormalization,MaxPooling3D,Flatten
from keras.layers.merge import _Merge
from keras import backend as K

# tensorboard log
from logger import Logger
import time
import datetime

#keras.backend.get_session().run(tf.global_variables_initializer())


#init = K.tf.global_variables_initializer() 
#K.get_session().run(init)

from skimage.util.shape import view_as_blocks


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--arch", dest="arch", help="Unet", default=None, type="string")
parser.add_option("--init", dest="init", help="random | finetune | pre-trained", default=None, type="string")
parser.add_option("--backbone", dest="backbone", help="the backbones", default=None, type="string")
parser.add_option("--decoder", dest="decoder_block_type", help="transpose | upsampling", default="upsampling", type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
parser.add_option("--nb_class", dest="nb_class", help="number of class", default=1, type="int")
parser.add_option("--verbose", dest="verbose", help="verbose", default=0, type="int")
parser.add_option("--weights", dest="weights", help="pre-trained encoder weights", default=None, type="string")
parser.add_option("--gweights", dest="gweights", help="pre-trained generator weights", default=None, type="string")
parser.add_option("--dweights", dest="dweights", help="pre-trained discriminator weights", default=None, type="string")
parser.add_option("--note", dest="note", help="notes of experiment setup", default="", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type="int")
parser.add_option("--scale", dest="scale", help="the scale of pre-trained data", default=32, type="int")
parser.add_option("--workers", dest="workers", help="number of workers", default=8, type="int")
parser.add_option('--resume_iters', dest="resume_iters", type=int, default=None, help='resume training from this step')
parser.add_option('--loss', dest="loss", type="string", default=None, help='L1/L2 loss')
parser.add_option('--adv', dest="adv", action='store_true', default=True, help='adv learning')
parser.add_option('--clc_head', dest="clc_head", type="string", default="conv", help='conv|fully_connect')
parser.add_option('--exp_part', dest="exp_part", type="string", default="encoder", help='encoder|en-decoder|en-decoder-adv')

parser.add_option("--rescale_rate", dest="rescale_rate", help="chance to perform rescaling", default=0.0, type="float")
parser.add_option("--nonlinear_rate", dest="nonlinear_rate", help="chance to perform nonlinear", default=0.9, type="float")
parser.add_option("--denoise_rate", dest="denoise_rate", help="chance to perform denoise", default=0.0, type="float")
parser.add_option("--paint_rate", dest="paint_rate", help="chance to perform painting", default=0.9, type="float")
parser.add_option("--outpaint_rate", dest="outpaint_rate", help="chance to perform out-painting", default=0.8, type="float")
parser.add_option("--rotation_rate", dest="rotation_rate", help="chance to perform rotation", default=0.0, type="float")
parser.add_option("--flip_rate", dest="flip_rate", help="chance to perform flipping", default=0.9, type="float")
parser.add_option("--local_rate", dest="local_rate", help="chance to perform local shuffle pixel", default=0.0, type="float")

(options, args) = parser.parse_args()

assert options.backbone in [None,
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
                        'pre-trained',
                       ]
assert options.decoder_block_type in ['transpose',
                                      'upsampling'
                                     ]

seed = 1
random.seed(seed)
model_path = os.path.join("Models/exp_rubik++/", options.exp_part)
if not os.path.exists(model_path):
    os.makedirs(model_path) 
    
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
class setup_config():
    # DATA_DIR = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes_64x64x64"
    DATA_DIR = "/home/zguo32/DATA/Self_Learning_Cubes_64x64x64/"
    optimizer = "Adam"
    nb_epoch = 10000
    patience = 40
    lr = 1e-4
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
                 rotation_rate=0.0,
                 flip_rate=0.0,
                 local_rate=0.9,
                 verbose=1,
                 resume_iters=None,
                 workers=1,
                 loss='L1',
                 adv=True,
                ):
        self.model = model
        self.backbone = backbone
        self.init = init
        if backbone is not None:
            self.exp_name = model + "-" + backbone + "-" + note
        else:
            self.exp_name = model + "-" + note
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
        self.resume_iters = resume_iters
        self.workers = workers
        self.loss = loss
        self.adv = False
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"
        if self.init != "pre-trained":
            self.weights = None
        else:
            self.weights = "Models/exp_rubic/Unet-resnet18-3drubik-32-3.h5"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

config = setup_config(model=options.arch,
                      backbone=options.backbone,
                      init=options.init,
                      note=options.note,
                      decoder_block_type=options.decoder_block_type,
                      input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      batch_size=options.batch_size,
                      nb_class=options.nb_class,
                      verbose=options.verbose,
                      rescale_rate=options.rescale_rate,
                      nonlinear_rate=options.nonlinear_rate,
                      denoise_rate=options.denoise_rate,
                      paint_rate=options.paint_rate,
                      outpaint_rate=options.outpaint_rate,
                      rotation_rate=options.rotation_rate,
                      flip_rate=options.flip_rate,
                      local_rate=options.local_rate,
                      resume_iters = options.resume_iters,
                      workers = options.workers,
                      loss = options.loss,
                      adv = options.adv
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

def rotate(matrix, degree):
    out = np.rot90(matrix,k=degree%4,axes=(2,1))
    if degree > 4:
        out = np.rot90(out,k=2,axes=(0,1))
    return out

        
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            return next(self.it)
        
def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g
        
@threadsafe_generator        
def randrot(data,batch_size,exp_choice):
    #input shape [B,1,H,W,D]
    #H must equal to W
    print(data.shape)
    H, W, D = data.shape[2], data.shape[3], data.shape[4]    
    while True:
        # random sample images
        batch_bag = data[random.sample(range(data.shape[0]),batch_size)]
        ori_bag = copy.deepcopy(batch_bag)
                
                
        #rotation labels 0-->7, corresponding to (0,0),(0,90),(0,180),(0,270),(180,0),(180,90),(180,180),(180,270)
        labels = np.random.randint(8,size=batch_size)
        x = []
        for idx, label in enumerate(labels):
            x.append(rotate(batch_bag[idx],label))

        x = np.asarray(x)
        labels = np.eye(8)[labels]
        labels = np.asarray(labels)
        #print(x.shape,labels.shape)

        if exp_choice == 'encoder':
            yield(x,labels)
        else:
            yield (x, [ori_bag, labels])
    
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
            gen_img = fake_img[b,0,:,:,c]
            ori_img = y_recont[b,0,:,:,c]
            disr_img = x[b,0,:,:,c]
            
            #difference map ==> normalize
            diff_img = gen_img - ori_img
            diff_img = (diff_img - np.min(diff_img))/(np.max(diff_img) - np.min(diff_img))                                   
                
            temp = np.concatenate((disr_img,blank_ver,gen_img,blank_ver,ori_img,blank_ver,diff_img),axis=1)
            sample_img.extend(temp)
            sample_img.extend(blank_hor)

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

  
# learning rate schedule
# source: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch):
    
    initial_lrate = config.lr
    drop = 0.5
    epochs_drop = int(config.patience * 0.8)
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
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
    
gen_base = unet_model_3d((1, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True)
dis_base = unet_model_3d((2, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True)

encoder_out = gen_base.get_layer('depth_7_relu').output

if options.clc_head == 'conv':
    encoder_out = Conv3D(512, (3,3,3), padding='same', strides=(1,1,1), name="jigsaw_conv")(encoder_out)
    encoder_out = BatchNormalization(axis=1, name="jigsaw_bn")(encoder_out)
    encoder_out = Activation('relu', name="jigsaw_relu")(encoder_out)
    cls_conv2=GlobalAveragePooling3D()(encoder_out)
    cls_fc = Dense(1024, activation="relu", name='cls_fc')(cls_conv2)
    cls_output = Dense(8, activation="softmax", name='cls_output')(cls_fc)
elif options.clc_head == 'fully_connect':
    encoder_out = Flatten()(encoder_out)
    # encoder_out = GlobalAveragePooling3D()(encoder_out)
    encoder_out = Dense(2048,activation="relu", name="clc_dense_1")(encoder_out)
    encoder_out = BatchNormalization(axis=1, name="clc_bn_1")(encoder_out)
    encoder_out = Dropout(rate=0.5,name="clc_dp_1")(encoder_out)
    encoder_out = Dense(1024,activation="relu", name="clc_dense_2")(encoder_out)
    encoder_out = BatchNormalization(axis=1, name="clc_bn_2")(encoder_out)
    encoder_out = Dropout(rate=0.5,name="clc_dp_2")(encoder_out)    
    cls_output = Dense(8, activation="softmax", name='cls_output')(encoder_out)
else:
    # encoder_out = Flatten()(encoder_out)
    encoder_out = GlobalAveragePooling3D()(encoder_out)
    encoder_out = Dense(2048,activation="relu", name="clc_dense_1")(encoder_out)    
    cls_output = Dense(8, activation="softmax", name='cls_output')(encoder_out)
    
    
genopt = keras.optimizers.Nadam(lr=1e-4, beta_1=0.5, beta_2=0.999)
lrate = keras.callbacks.LearningRateScheduler(step_decay, verbose=1)

if options.exp_part == "encoder":
    gen_base = keras.models.Model(gen_base.input,cls_output)
    gen_base.compile(optimizer = genopt,
                loss="categorical_crossentropy",
                metrics=['accuracy'])
        
else:
    gen_base = keras.models.Model(gen_base.input,[gen_base.output,cls_output])
    if config.init == 'pre-trained':
        if options.gweights is not None:
            print("Load the pre-trained generator weights from {}".format(options.gweights))
            gen_base.load_weights(options.gweights)
        else:
            pre_trained_model = keras.models.load_model(options.weights)
            for i in range(28):
                gen_base.layers[i].set_weights(pre_trained_model.layers[i].get_weights())     
            
    if config.loss == 'L1':
        gen_base.compile(optimizer = genopt,
                loss=["mae", "categorical_crossentropy"],
                loss_weights=[100, 1],
                metrics={'activation_1':"mae",'cls_output':"accuracy"})
    else:
        gen_base.compile(optimizer = genopt,
                loss=["mse", "categorical_crossentropy"],
                loss_weights=[100, 1],
                metrics={'activation_1':"mse",'cls_output':"accuracy"})
        
        #building discriminator
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
    
    disopt = keras.optimizers.Nadam(lr=1e-4, beta_1=0.5, beta_2=0.999)
    
    dis.compile(optimizer=disopt,
                loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                loss_weights=[1, 1, 10])
    
    
                
    #building generator            
    for layer in dis_model.layers:
        layer.trainable = False
    dis_model.trainable = False
    
    
    disarrange_inp = Input(shape=(1,config.input_rows,config.input_cols,config.input_deps))
    [gen_out,cls_out] = gen_base(disarrange_inp)
    
    x = Concatenate(axis=1)([gen_out,disarrange_inp])           
    real_fake = dis_model(x)
    
    gen = keras.models.Model(disarrange_inp, [gen_out, cls_out, real_fake])
                                                                    
    if config.loss == 'L1':
        gen.compile(optimizer = genopt,
                loss=["mae", "categorical_crossentropy", wasserstein_loss],
                loss_weights=[100, 1, 1])
    else:
        gen.compile(optimizer = genopt,
                loss=["mse", "categorical_crossentropy", wasserstein_loss],
                loss_weights=[100, 1, 1])
    
    for layer in dis_model.layers:
        layer.trainable = True            
    dis_model.trainable = True

gen_base.summary()



x_train = []
for i,fold in enumerate(tqdm(config.train_fold+config.test_fold)):
# for i,fold in enumerate(tqdm([0])):
    #s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x32_"+str(fold)+".npy"))
    s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x64_"+str(fold)+".npy"),mmap_mode='r')
    x_train.extend(s)
#x_train = np.asarray(x_train)
x_train = np.expand_dims(np.array(x_train), axis=1)
print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))

x_valid = []
for i,fold in enumerate(tqdm(config.valid_fold)):
# for i,fold in enumerate(tqdm([6])):
    #s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x32_"+str(fold)+".npy"))
    s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x64_"+str(fold)+".npy"),mmap_mode='r')
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)
#x_valid = np.asarray(x_valid)
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

logger = Logger(os.path.join(logs_path, config.exp_name))

train_data_gen = randrot(x_train, config.batch_size, options.exp_part)
validation_data_gen = randrot(x_valid, config.batch_size, options.exp_part)

valid = -np.ones((config.batch_size, 1, 8, 8, 8))
fake =  np.ones((config.batch_size, 1, 8, 8, 8))
dummy = np.zeros((config.batch_size, 1, 8, 8, 8)) # Dummy gt for gradient penalty


def restore_model(exp_path, config, options):
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(config.resume_iters))
    print(exp_path)
    if options.exp_part == 'en-decoder-adv':
        G_path = os.path.join(exp_path, config.exp_name+'-{}-G.h5'.format(config.resume_iters))
        D_path = os.path.join(exp_path, config.exp_name+'-{}-D.h5'.format(config.resume_iters))
        dis_model.load_weights(D_path)
    else:
        G_path = os.path.join(model_path, config.exp_name+".h5")
    gen_base.load_weights(G_path)
    # 


start_time = time.time()
total_iter = x_train.shape[0]//(config.batch_size)

best_mse = 1.0

start_ep = 0
if config.resume_iters is not None:
    start_ep = config.resume_iters
    restore_model(exp_path, config,options) 

if not config.adv:
    
    if os.path.exists(os.path.join(model_path, config.exp_name+".txt")):
        os.remove(os.path.join(model_path, config.exp_name+".txt"))
    with open(os.path.join(model_path, config.exp_name+".txt"),'w') as fh:
        gen_base.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

    shutil.rmtree(os.path.join(logs_path, config.exp_name), ignore_errors=True)
    if not os.path.exists(os.path.join(logs_path, config.exp_name)):
        os.makedirs(os.path.join(logs_path, config.exp_name))
    tbCallBack = TensorBoard(log_dir=os.path.join(logs_path, config.exp_name),
                         histogram_freq=0, # histogram_freq=0
                         write_graph=True, 
                         write_images=True,
                        )
    tbCallBack.set_model(gen_base)
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=config.patience, 
                                               verbose=0,
                                               mode='min',
                                              )
    check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, config.exp_name+".h5"),
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only=True, 
                                              mode='min',
                                             )
    def testmodel(epoch,logs):
        if epoch%2 == 0:
            predx, [ predy , _ ] = next(train_data_gen)
            predout,_ = gen_base.predict(predx)
            save_image(predx,predy,predout,epoch)
            
    if options.exp_part == 'encoder':
        callbacks = [check_point, early_stopping, tbCallBack, lrate]
    else:
        saving_image = keras.callbacks.LambdaCallback(on_epoch_end=testmodel)            
        callbacks = [check_point, early_stopping, tbCallBack, lrate, saving_image]

        
    
    gen_base.fit_generator(train_data_gen,#generate_pair(x_train, config.batch_size),train_data,
                            validation_data=validation_data_gen, #validation_data=generate_pair(x_valid, config.batch_size), valid_data
                            validation_steps=int(x_valid.shape[0]//config.batch_size),
                            steps_per_epoch=x_train.shape[0]//config.batch_size, 
                            epochs=config.nb_epoch,
                            max_queue_size=config.workers+2, #6
                            workers=config.workers, # comment this if error
                            use_multiprocessing=False, 
                            shuffle=True,
                            verbose=config.verbose, 
                            callbacks=callbacks,
                            initial_epoch=start_ep
                           )
    
    
    
    
    
    
    
    
    

else:
    train_q = GeneratorEnqueuer(train_data_gen, use_multiprocessing=False)
    train_q.start(workers=config.workers, max_queue_size=config.workers*2)
    # else:
        # train_q = GeneratorEnqueuer(train_data_gen, use_multiprocessing=False)
        # train_q.start()
    train_data = train_q.get()


    # training starts
    for ep in range(start_ep, config.nb_epoch):
        print("Epoch", ep+1, "/", config.nb_epoch)
        
        # val_loss = gen_base.evaluate_generator(validation_data_gen, steps=int(x_valid.shape[0]//config.batch_size), verbose=1)
        # print("Epoch {} validation MSE = {:.5f}".format(ep+1, val_loss[2]))
        
        for it in range(total_iter):
            x, [y, l] = next(train_data)
            fake_img, _ = gen_base.predict(x)
            # print(fake_img.shape)
            #print(x.shape)
    
            dis_loss = dis.train_on_batch([y, fake_img, x], [valid, fake, dummy])
            if it%5==0:
                gen_loss = gen.train_on_batch(x, [y, l, valid])
    
        
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
                log = "Elapsed [{}], Epoch [{}], Iteration [{}/{}]".format(et, ep+1, it, total_iter)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
    
                for tag, value in loss.items():
                    logger.scalar_summary(tag, value, it)
            
        if (ep+1) % 2 == 0:
            print(fake_img.shape)
            # print(type(x),type(y))
            diff = x!=y
            same = x==y
            A = y[diff]
            A_Hat = fake_img[diff]
            B = y[same]
            B_Hat = fake_img[same]
            
            mse_diff = mse(A,A_Hat)
            mse_same = mse(B,B_Hat)        
            
            blank_ver = np.ones((fake_img.shape[2],10))
            blank_hor = np.ones((10, fake_img.shape[2]*4+30))
            sample_img = []
            sample_img.extend(blank_hor)
            sample_img.extend(blank_hor)        
            for b in range(fake_img.shape[0]):
                for c in range(6):
                    temp_axis = [slice(None)]*fake_img.ndim
                    # for c in [0,1] ==> dim 4, [2,3]==> dim 3, [4,5] ==> dim 2, alternating from 0 and -1 ==> showing the
                    # images from all 6 faces
                    # temp_axis[4-c//2] = -1*(c%2)
                    temp_axis[4-c//2] = -7*(c%2)+3 # alternating from -4 to 3
                    temp_axis[0] = b
                    temp_axis[1] = 0
                    gen_img = fake_img[tuple(temp_axis)]
                    ori_img = y[tuple(temp_axis)]
                    disr_img = x[tuple(temp_axis)]
                    
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
            
            
            sample_img_save.save(sample_path)
            print('Saved real and fake images into {}...'.format(sample_path))
        
        # Decay learning rates.
        # if (ep+1) % 5 == 0 and (ep+1) > 100:
            # g_lr -= (0.0001 / float(1000))
            # d_lr -= (0.0001 / float(1000))
            # self.update_lr(g_lr, d_lr)
            # print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
                    
        
        
        # validation
        val_loss = gen_base.evaluate_generator(validation_data_gen, steps=int(x_valid.shape[0]//config.batch_size), verbose=1)
        print("Epoch {} validation MSE = {:.5f}".format(ep+1, val_loss[2]))
    
        if best_mse > val_loss[2]:
            dis_model.save_weights(os.path.join(exp_path,config.exp_name+'-{}-D.h5'.format(ep+1)))
            gen_base.save_weights(os.path.join(exp_path, config.exp_name+'-{}-G.h5'.format(ep+1)))
            best_mse = val_loss[2]



                                                                                                                                                                                                                                                                                         