#!/usr/bin/env python
# -*- coding: utf-8 -*-


import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD

import numpy as np

#dropout_rate = 0.5

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def BasicBlock(input_tensor, stage, nb_filter, kernel_size=3):

    x = keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size),
                            name='conv'+stage+'_1', 
                            kernel_initializer='he_normal', 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4))(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn'+stage+'_1')(x)
    x = keras.layers.ReLU(name='relu'+stage+'_1')(x)
    #x = keras.layers.core.Dropout(dropout_rate, name='dp'+stage+'_1')(x)

    x = keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), 
                            name='conv'+stage+'_2', 
                            kernel_initializer='he_normal', 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn'+stage+'_2')(x)
    x = keras.layers.ReLU(name='relu'+stage+'_2')(x)
    #x = keras.layers.core.Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

def BasicBlockV(input_tensor, stage, nb_filter, kernel_size=3):

    x = keras.layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size),
                            name='conv'+stage+'_1', 
                            kernel_initializer='he_normal', 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4))(input_tensor)
    # x = keras.layers.BatchNormalization(axis=bn_axis, name='bn'+stage+'_1')(x)
    x = keras.layers.ReLU(name='relu'+stage+'_1')(x)
    #x = keras.layers.core.Dropout(dropout_rate, name='dp'+stage+'_1')(x)

    x = keras.layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), 
                            name='conv'+stage+'_2', 
                            kernel_initializer='he_normal', 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = keras.layers.BatchNormalization(axis=bn_axis, name='bn'+stage+'_2')(x)
    x = keras.layers.ReLU(name='relu'+stage+'_2')(x)
    #x = keras.layers.core.Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

########################################

"""
Standard U-Net [Ronneberger et.al., 2015]
Total params: 7,771,297
"""
def U_Net(img_rows, img_cols, color_type=1, num_class=1, activate='sigmoid'):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = keras.layers.Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = keras.layers.Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = BasicBlock(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = BasicBlock(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = BasicBlock(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = BasicBlock(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = BasicBlock(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = keras.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = keras.layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = BasicBlock(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = keras.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = keras.layers.concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = BasicBlock(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = keras.layers.concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = BasicBlock(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = keras.layers.concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = BasicBlock(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = keras.layers.Conv2D(num_class, (1, 1), 
                                      activation=activate, 
                                      name='output', 
                                      kernel_initializer='he_normal', 
                                      padding='same', 
                                      kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_5)

    model = keras.models.Model(input=img_input, output=unet_output)

    return model

def V_Net(img_rows, img_cols, img_deps, num_class=1, activate='sigmoid'):

    # nb_filter = [32,64,128,256,512]
    nb_filter = [16,32,64,128,256]
    # nb_filter = [16,16,16,16,16]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        img_input = keras.layers.Input(shape=(img_rows, img_cols, img_deps, 1), name='main_input')
        bn_axis = 4
    else:
        img_input = keras.layers.Input(shape=(1, img_deps, img_rows, img_cols), name='main_input')
        bn_axis = 1

    conv1_1 = BasicBlockV(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = keras.layers.pooling.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = BasicBlockV(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = keras.layers.pooling.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    conv3_1 = BasicBlockV(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = keras.layers.pooling.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    conv4_1 = BasicBlockV(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = keras.layers.pooling.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool4')(conv4_1)

    conv5_1 = BasicBlockV(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = keras.layers.Conv3DTranspose(nb_filter[3], (2, 2, 2), strides=(2, 2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = keras.layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = BasicBlockV(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = keras.layers.Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = keras.layers.concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = BasicBlockV(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = keras.layers.Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = keras.layers.concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = BasicBlockV(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = keras.layers.Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = keras.layers.concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = BasicBlockV(conv1_5, stage='15', nb_filter=nb_filter[0])

    vnet_output = keras.layers.Conv3D(num_class, (1, 1, 1), 
                                      activation=activate, 
                                      name='output', 
                                      kernel_initializer='he_normal', 
                                      padding='same', 
                                      kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_5)

    model = keras.models.Model(input=img_input, output=vnet_output)

    return model


"""
wU-Net for comparison
Total params: 9,295,126
"""
def wU_Net(img_rows, img_cols, color_type=1, num_class=1, activate='sigmoid'):

    # nb_filter = [32,64,128,256,512]
    nb_filter = [35,70,140,280,560]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = keras.layers.Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = keras.layers.Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = BasicBlock(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = BasicBlock(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = BasicBlock(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = BasicBlock(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = BasicBlock(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = keras.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = keras.layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = BasicBlock(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = keras.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = keras.layers.concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = BasicBlock(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = keras.layers.concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = BasicBlock(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = keras.layers.concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = BasicBlock(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = keras.layers.Conv2D(num_class, (1, 1), 
                                      activation=activate, 
                                      name='output', 
                                      kernel_initializer='he_normal', 
                                      padding='same', 
                                      kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_5)

    model = keras.models.Model(input=img_input, output=unet_output)

    return model

"""
Standard UNet++ [Zhou et.al., 2018]
Total params: 9,056,193
"""
def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False, merge_type="concatenate", activate='sigmoid'):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = keras.layers.Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = keras.layers.Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = BasicBlock(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = BasicBlock(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    #conv1_2 = keras.layers.concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = keras.layers.Add(name='merge12')([up1_2, conv1_1])
    conv1_2 = BasicBlock(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = BasicBlock(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    #conv2_2 = keras.layers.concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = keras.layers.Add(name='merge22')([up2_2, conv2_1])
    conv2_2 = BasicBlock(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    #conv1_3 = keras.layers.concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = keras.layers.Add(name='merge13')([up1_3, conv1_1, conv1_2])
    conv1_3 = BasicBlock(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = BasicBlock(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = keras.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    #conv3_2 = keras.layers.concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = keras.layers.Add(name='merge32')([up3_2, conv3_1])
    conv3_2 = BasicBlock(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    #conv2_3 = keras.layers.concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = keras.layers.Add(name='merge23')([up2_3, conv2_1, conv2_2])
    conv2_3 = BasicBlock(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    #conv1_4 = keras.layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = keras.layers.Add(name='merge14')([up1_4, conv1_1, conv1_2, conv1_3])
    conv1_4 = BasicBlock(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = BasicBlock(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = keras.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    #conv4_2 = keras.layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = keras.layers.Add(name='merge42')([up4_2, conv4_1])
    conv4_2 = BasicBlock(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = keras.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    #conv3_3 = keras.layers.concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = keras.layers.Add(name='merge33')([up3_3, conv3_1, conv3_2])
    conv3_3 = BasicBlock(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    #conv2_4 = keras.layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = keras.layers.Add(name='merge24')([up2_4, conv2_1, conv2_2, conv2_3])
    conv2_4 = BasicBlock(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    #conv1_5 = keras.layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = keras.layers.Add(name='merge15')([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
    conv1_5 = BasicBlock(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_1', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_2)
    nestnet_output_2 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_2', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_3)
    nestnet_output_3 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_3', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_4)
    nestnet_output_4 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_4', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_5)

    if deep_supervision:
        model = keras.models.Model(input=img_input, output=[nestnet_output_1,
                                                           nestnet_output_2,
                                                           nestnet_output_3,
                                                           nestnet_output_4])
    else:
        model = keras.models.Model(input=img_input, output=[nestnet_output_4])

    return model
   
"""
Deep Layer Aggregation [Yu et.al., 2018]
Total params: 8,742,849
"""
def DLA(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False, activate='sigmoid'):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = keras.layers.Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = keras.layers.Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = BasicBlock(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = BasicBlock(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = keras.layers.concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = BasicBlock(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = BasicBlock(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = keras.layers.concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = BasicBlock(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = keras.layers.concatenate([up1_3, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = BasicBlock(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = BasicBlock(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = keras.layers.pooling.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = keras.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = keras.layers.concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = BasicBlock(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = keras.layers.concatenate([up2_3, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = BasicBlock(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = keras.layers.concatenate([up1_4, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = BasicBlock(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = BasicBlock(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = keras.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = keras.layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = BasicBlock(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = keras.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = keras.layers.concatenate([up3_3, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = BasicBlock(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = keras.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = keras.layers.concatenate([up2_4, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = BasicBlock(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = keras.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = keras.layers.concatenate([up1_5, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = BasicBlock(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_1', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_2)
    nestnet_output_2 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_2', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_3)
    nestnet_output_3 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_3', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_4)
    nestnet_output_4 = keras.layers.Conv2D(num_class, (1, 1), 
                                           activation=activate, 
                                           name='output_4', 
                                           kernel_initializer='he_normal', 
                                           padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(1e-4))(conv1_5)

    if deep_supervision:
        model = keras.models.Model(input=img_input, output=[nestnet_output_1,
                                                           nestnet_output_2,
                                                           nestnet_output_3,
                                                           nestnet_output_4])
    else:
        model = keras.models.Model(input=img_input, output=[nestnet_output_4])

    return model

"""
FCN8s with VGG16 as backbone [Long et.al., 2015]
Total params: 134,265,699
"""
def FCN8s(img_rows=224, img_cols=224, color_type=3, num_class=1, weights='imagenet', activate='sigmoid'):

    vgg16 = keras.applications.vgg16.VGG16(include_top=False,
                                           weights=weights,
                                           input_tensor=None,
                                           input_shape=(img_rows, img_cols, color_type),
                                           )

    ip = Input(shape=(img_rows, img_cols, color_type), name="main_input")
    h = vgg16.layers[1](ip)
    h = vgg16.layers[2](h)
    h = vgg16.layers[3](h)
    h = vgg16.layers[4](h)
    h = vgg16.layers[5](h)
    h = vgg16.layers[6](h)
    h = vgg16.layers[7](h)
    h = vgg16.layers[8](h)
    h = vgg16.layers[9](h)
    h = vgg16.layers[10](h)

    # split layer
    p3 = h

    h = vgg16.layers[11](h)
    h = vgg16.layers[12](h)
    h = vgg16.layers[13](h)
    h = vgg16.layers[14](h)

    # split layer
    p4 = h

    h = vgg16.layers[15](h)
    h = vgg16.layers[16](h)
    h = vgg16.layers[17](h)
    h = vgg16.layers[18](h)

    p5 = h


    # vgg16 ends
    # FCN8s starts
    # Reference: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py


    relu6 = keras.layers.Conv2D(4096, 7, padding='same', activation='relu', name='fc6', kernel_regularizer=keras.regularizers.l2(1e-4))(p5)
    drop6 = keras.layers.core.Dropout(0.5, name='drop6')(relu6)
    relu7 = keras.layers.Conv2D(4096, 1, padding='same', activation='relu', name='fc7', kernel_regularizer=keras.regularizers.l2(1e-4))(drop6)
    drop7 = keras.layers.core.Dropout(0.5, name='drop7')(relu7)
    score_fr = keras.layers.Conv2D(num_class, 1, padding='same', name='score_fr', kernel_regularizer=keras.regularizers.l2(1e-4))(drop7)
    upscore2 = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='upscore2')(score_fr)

    score_pool4 = keras.layers.Conv2D(num_class, 1, padding='same', name='score_pool4', kernel_regularizer=keras.regularizers.l2(1e-4))(p4)
    fuse_pool4 = keras.layers.Add(name='fuse_pool4')([upscore2, score_pool4])
    upscore_pool4 = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='upscore_pool4')(fuse_pool4)

    score_pool3 = keras.layers.Conv2D(num_class, 1, padding='same', name='score_pool3', kernel_regularizer=keras.regularizers.l2(1e-4))(p3)
    fuse_pool3 = keras.layers.Add(name='fuse_pool3')([upscore_pool4, score_pool3])
    upscore8 = keras.layers.Conv2DTranspose(num_class, 16, strides=8, padding='same', activation=activate, use_bias=False, name='upscore8')(fuse_pool3)

    # h = Softmax(name='h')(upscore8) # not nomalized
    
    model = keras.models.Model(ip, upscore8)

    return model

"""
Improve FCN with VGG16 as backbone using nested style
Total params: 134,286,225
"""
def FCN_NestNet(img_rows=224, img_cols=224, color_type=3, num_class=1, activate="sigmoid"):

    vgg16 = keras.applications.vgg16.VGG16(include_top=False,
                                           weights='imagenet',
                                           input_tensor=None,
                                           input_shape=(img_rows, img_cols, color_type),
                                           )

    ip = Input(shape=(img_rows, img_cols, color_type))
    h = vgg16.layers[1](ip)
    h = vgg16.layers[2](h)
    h = vgg16.layers[3](h)

    # split layer
    p1 = h

    h = vgg16.layers[4](h)
    h = vgg16.layers[5](h)
    h = vgg16.layers[6](h)

    # split layer
    p2 = h 

    h = vgg16.layers[7](h)
    h = vgg16.layers[8](h)
    h = vgg16.layers[9](h)

    up1_1_score = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='up1_1_score')(h)
    up1_1_conv = keras.layers.Conv2D(num_class, 1, padding='same', name='up1_1_conv', kernel_regularizer=keras.regularizers.l2(1e-4))(p1)
    up1_1_conv_fuse = keras.layers.Add(name='up1_1_conv_fuse')([up1_1_score, up1_1_conv])

    h = vgg16.layers[10](h)

    # split layer
    p3 = h

    h = vgg16.layers[11](h)
    h = vgg16.layers[12](h)
    h = vgg16.layers[13](h)

    up2_1_score = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='up2_1_score')(h)
    up2_1_conv = keras.layers.Conv2D(num_class, 1, padding='same', name='up2_1_conv', kernel_regularizer=keras.regularizers.l2(1e-4))(p2)
    up2_1_conv_fuse = keras.layers.Add(name='up2_1_conv_fuse')([up2_1_score, up2_1_conv])

    up1_2_score = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='up1_2_score')(up2_1_conv_fuse)
    up1_2_conv = keras.layers.Conv2D(num_class, 1, padding='same', name='up1_2_conv', kernel_regularizer=keras.regularizers.l2(1e-4))(up1_1_conv_fuse)
    up1_2_conv_fuse = keras.layers.Add(name='up1_2_conv_fuse')([up1_2_score, up1_2_conv, up1_1_conv])

    h = vgg16.layers[14](h)

    # split layer
    p4 = h

    h = vgg16.layers[15](h)
    h = vgg16.layers[16](h)
    h = vgg16.layers[17](h)

    up3_1_score = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='up3_1_score')(h)
    up3_1_conv = keras.layers.Conv2D(num_class, 1, padding='same', name='up3_1_conv', kernel_regularizer=keras.regularizers.l2(1e-4))(p3)
    up3_1_conv_fuse = keras.layers.Add(name='up3_1_conv_fuse')([up3_1_score, up3_1_conv])

    up2_2_score = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='up2_2_score')(up3_1_conv_fuse)
    up2_2_conv = keras.layers.Conv2D(num_class, 1, padding='same', name='up2_2_conv', kernel_regularizer=keras.regularizers.l2(1e-4))(up2_1_conv_fuse)
    up2_2_conv_fuse = keras.layers.Add(name='up2_2_conv_fuse')([up2_2_score, up2_2_conv, up2_1_conv])

    up1_3_score = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='up1_3_score')(up2_2_conv_fuse)
    up1_3_conv = keras.layers.Conv2D(num_class, 1, padding='same', name='up1_3_conv', kernel_regularizer=keras.regularizers.l2(1e-4))(up1_2_conv_fuse)
    up1_3_conv_fuse = keras.layers.Add(name='up1_3_conv_fuse')([up1_3_score, up1_3_conv, up1_2_conv, up1_1_conv])
    
    h = vgg16.layers[18](h)

    p5 = h

    # vgg16 ends
    # FCN2s starts

    # 32s
    relu6 = keras.layers.Conv2D(4096, 7, padding='same', activation='relu', name='fc6', kernel_regularizer=keras.regularizers.l2(1e-4))(p5)
    drop6 = Dropout(0.5, name='drop6')(relu6)
    relu7 = keras.layers.Conv2D(4096, 1, padding='same', activation='relu', name='fc7', kernel_regularizer=keras.regularizers.l2(1e-4))(drop6)
    drop7 = Dropout(0.5, name='drop7')(relu7)
    score_fr = keras.layers.Conv2D(num_class, 1, padding='same', name='score_fr', kernel_regularizer=keras.regularizers.l2(1e-4))(drop7)
    upscore_pool5 = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='upscore_pool5')(score_fr)

    # 16s
    score_pool4 = keras.layers.Conv2D(num_class, 1, padding='same', name='score_pool4', kernel_regularizer=keras.regularizers.l2(1e-4))(p4)
    fuse_pool4 = keras.layers.Add(name='fuse_pool4')([upscore_pool5, score_pool4])
    upscore_pool4 = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='upscore_pool4')(fuse_pool4)

    # 8s
    score_pool3 = keras.layers.Conv2D(num_class, 1, padding='same', name='score_pool3', kernel_regularizer=keras.regularizers.l2(1e-4))(up3_1_conv_fuse)
    fuse_pool3 = keras.layers.Add(name='fuse_pool3')([upscore_pool4, score_pool3, up3_1_conv])
    upscore_pool3 = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='upscore_pool3')(fuse_pool3)

    # 4s
    score_pool2 = keras.layers.Conv2D(num_class, 1, padding='same', name='score_pool2', kernel_regularizer=keras.regularizers.l2(1e-4))(up2_2_conv_fuse)
    fuse_pool2 = keras.layers.Add(name='fuse_pool2')([upscore_pool3, score_pool2, up2_2_conv, up2_1_conv])
    upscore_pool2 = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', use_bias=False, name='upscore_pool2')(fuse_pool2)

    # 2s
    score_pool1 = keras.layers.Conv2D(num_class, 1, padding='same', name='score_pool1', kernel_regularizer=keras.regularizers.l2(1e-4))(up1_3_conv_fuse)
    fuse_pool1 = keras.layers.Add(name='fuse_pool1')([upscore_pool2, score_pool1, up1_3_conv, up1_2_conv, up1_1_conv])
    upscore2 = keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', activation=activate, use_bias=False, name='upscore2')(fuse_pool1)

    model = keras.models.Model(ip, upscore2)

    return model



if __name__ == '__main__':
    
    model = U_Net(96,96,1)
    model.summary()

    model = wU_Net(96,96,1)
    model.summary()

    model = Nest_Net(96,96,1)
    model.summary()

    model = DLA(96,96,1)
    model.summary()
