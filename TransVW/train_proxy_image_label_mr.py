"""
    Owner: Fatemeh Haghighi, Mohammad Reza Hosseinzadeh Taher
    Copyright  to ASU JLiang Lab
"""

#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

"""
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 python -W ignore train_proxy_image_label_mr.py \
--explanation mr1\

"""



import warnings

warnings.filterwarnings('ignore')
import os
import keras

print("Keras = {}".format(keras.__version__))
import tensorflow as tf

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
from keras.layers import GlobalAveragePooling3D, Dense,Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D,Concatenate
import random
from sklearn.utils import shuffle
from unet3d import *
from keras.callbacks import LambdaCallback, TensorBoard
from glob import glob
from skimage.transform import resize
from optparse import OptionParser
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
# from segmentation_models import Nestnet, Unet, Xnet
from keras.utils import plot_model
from datetime import datetime
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--arch", dest="arch", help="Unet", default="Unet", type="string")
parser.add_option("--decoder", dest="decoder_block_type", help="transpose | upsampling", default="upsampling",
                  type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type=int)
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type=int)
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type=int)
parser.add_option("--verbose", dest="verbose", help="verbose", default=1, type=int)
parser.add_option("--weights", dest="weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--unet_weights", dest="unet_weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=16, type=int)
parser.add_option("--run", dest="run", help="run number", default=1, type=int)
parser.add_option("--explanation", dest="explanation", help="explanation", default=None)
parser.add_option("--nb_classes", dest="nb_classes", help="nb_classes", default=44,type=int)
parser.add_option("--nb_patch_per_image", dest="nb_patch_per_image", type=int, default=3)
parser.add_option("--lr", dest="lr", type=float, default=0.01)

(options, args) = parser.parse_args()

assert options.arch in ['Unet',
                        'Nestnet',
                        'Xnet',
                        'Ynet'
                        ]

assert options.decoder_block_type in ['transpose',
                                      'upsampling'
                                      ]

seed = 1
random.seed(seed)
model_path = "Models/proxy_Img_label/"
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
    patience = 50


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
                 lr=0.01
                 ):
        self.model = model
        self.backbone = backbone
        self.exp_name = model + "img_label_"+explanation+"_lr"+str(lr)+"_"+str(run)
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.decoder_block_type = decoder_block_type
        self.nb_class = nb_class
        self.cls_classes=cls_classes
        self.lr=lr
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
                      lr=options.lr

                      )
config.display()

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


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images_dir,infos_dir ,batch_size=16, dim=(128, 128, 64),nb_classes=44):
        self.images = np.load(images_dir)
        self.infos=np.load(infos_dir)
        self.batch_size = batch_size
        self.dim = dim
        self.length = self.infos.shape[0]
        self.nb_classes=nb_classes

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

        input_rows = self.dim[0]
        input_cols = self.dim[1]
        input_depth = self.dim[2]
        x = np.zeros((self.batch_size, 1, input_rows, input_cols, input_depth), dtype="float")
        y_cls = np.zeros((self.batch_size, nb_classes), dtype="int32")
        count = 0
        elastic_prob=1
        aug_counter=3
        for i in range(file_list.shape[0]):
            patch = self.images[int(file_list[i,0]),:,:,:]
            label=int(file_list[i, 1])

            img_rows, img_cols, img_deps = patch.shape[0], patch.shape[1], patch.shape[2]
            y_cls[count, :] = keras.utils.to_categorical(label, self.nb_classes)



            patch_final = copy.deepcopy(patch)
            if i%aug_counter == 0:
                r=random.random()
                if r< 0.25:
                    patch_final = elastic_transform(patch_final)
                    # Local Shuffle Pixel
                # elif 0.25 <r<=0.5:
                #     image_temp = copy.deepcopy(patch_final)
                #     orig_image = copy.deepcopy(patch_final)
                #     num_block = 100
                #     for _ in range(num_block):
                #         block_noise_size_x, block_noise_size_y, block_noise_size_z = random.randint(1,
                #                                                                                     6), random.randint(
                #             1, 6), random.randint(1, 6)
                #         noise_x, noise_y, noise_z = random.randint(0, img_rows - block_noise_size_y), random.randint(0,
                #                                                                                                      img_cols - block_noise_size_x)
                #         ind_list = [i for i in range(block_noise_size_x * block_noise_size_y)]
                #         random.shuffle(ind_list)
                #         for order, shuff in enumerate(ind_list):
                #             image_temp[noise_x + order % block_noise_size_y, noise_y + order // block_noise_size_y,
                #             :] = orig_image[noise_x + shuff % block_noise_size_y, noise_y + shuff // block_noise_size_y,
                #                  :]
                #     patch_final = image_temp

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
                    patch_final = np.interp(patch_final, xvals, yvals)

                    # Inpainting
                elif 0.5 < r <= 0.75:
                    block_noise_size_x, block_noise_size_y, block_noise_size_z = random.randint(10, 20), random.randint(
                        10, 20), random.randint(10, 20)
                    noise_x, noise_y, noise_z = random.randint(3, img_rows - block_noise_size_x - 3), random.randint(3,
                                                                                                                     img_cols - block_noise_size_y - 3), random.randint(
                        3, img_deps - block_noise_size_z - 3)
                    patch_final[ noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,
                    noise_z:noise_z + block_noise_size_z] = random.random()
                # & Outpainting
                else:
                    block_noise_size_x, block_noise_size_y, block_noise_size_z = img_rows - random.randint(10,
                                                                                                           20), img_cols - random.randint(
                        10, 20), img_deps - random.randint(10, 20)
                    noise_x, noise_y, noise_z = random.randint(3, img_rows - block_noise_size_x - 3), random.randint(3,
                                                                                                                     img_cols - block_noise_size_y - 3), random.randint(
                        3, img_deps - block_noise_size_z - 3)
                    image_temp = copy.deepcopy(patch_final)
                    patch_final[:,:,:] = random.random()
                    patch_final[ noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,
                    noise_z:noise_z + block_noise_size_z] = image_temp[ noise_x:noise_x + block_noise_size_x,
                                                            noise_y:noise_y + block_noise_size_y,
                                                            noise_z:noise_z + block_noise_size_z]


            x[count,:, :,:,:] = np.expand_dims(patch_final, axis=0)
            count += 1

        x,y_cls = shuffle(x, y_cls, random_state=0)
        
        return x, y_cls



# learning rate schedule
# source: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch):
    initial_lrate = config.lr
    drop = 0.5
    epochs_drop = int(config.patience * 0.3)
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate

batch_size = int(config.batch_size)

if config.model == "Unet":
    base_model = unet_model_3d((1, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True)
    x = base_model.get_layer('depth_7_relu').output
    x = GlobalAveragePooling3D()(x)
    # x = Dense(1024, activation='relu')(x)
    output = Dense(config.cls_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=base_model.input, outputs=output)
if options.weights is not None:
    print("Load the pre-trained weights from {}".format(options.weights))
    model.load_weights(options.weights)
# model.compile(optimizer=keras.optimizers.SGD(lr=config.lr, momentum=0.9, decay=0.0, nesterov=False),
#               loss="categorical_crossentropy",
#               metrics=["accuracy", "categorical_crossentropy"])

model.compile(optimizer=keras.optimizers.Adam(lr=config.lr),
              loss="categorical_crossentropy",
              metrics=["accuracy", "categorical_crossentropy"])
model.summary()

training_generator = DataGenerator("./data/sims_trains_multires.npy",'./data/trains_infos_multires.npy',
                                       batch_size=batch_size,dim=(config.input_rows,config.input_cols,config.input_deps),nb_classes=options.nb_classes)
validation_generator = DataGenerator("./data/sims_validation_multires.npy",'./data/validation_infos_multires.npy',
                                         batch_size=batch_size,dim=(config.input_rows,config.input_cols,config.input_deps),nb_classes=options.nb_classes)

# plot_model(model, to_file=os.path.join(model_path, config.exp_name+".png"))
if os.path.exists(os.path.join(model_path, config.exp_name + ".txt")):
    os.remove(os.path.join(model_path, config.exp_name + ".txt"))
with open(os.path.join(model_path, config.exp_name + ".txt"), 'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(logs_path, config.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(logs_path, config.exp_name)):
    os.makedirs(os.path.join(logs_path, config.exp_name))

tbCallBack = TensorBoard(log_dir=os.path.join(logs_path, config.exp_name),
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True,
                        )
tbCallBack.set_model(model)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=config.patience,
                                               verbose=0,
                                               mode='min',
                                               )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, config.exp_name + ".h5"),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min',
                                              )
lrate = keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
callbacks = [check_point, early_stopping,tbCallBack, lrate]

# In[ ]:

# while config.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    # try:
model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=training_generator.length  // batch_size,
                            validation_steps=validation_generator.length  // batch_size,
                            epochs=config.nb_epoch,
                            # max_queue_size=4,
                            # workers=2,
                            use_multiprocessing=False,
                            shuffle=False,
                            verbose=config.verbose,
                            callbacks=callbacks,
                            )
    #     break
    # except tf.errors.ResourceExhaustedError as e:
    #     config.batch_size = int(config.batch_size / 2.0)
    #     print("\n> Batch size = {}".format(config.batch_size))

# max_queue_size=20, workers=4, use_multiprocessing=True, shuffle=True,
