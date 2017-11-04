import numpy as np
from keras.engine import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, add, Conv3DTranspose
from keras.optimizers import Adam
# from keras_contrib.layers.convolutional import Deconvolution3D
from keras import backend as K
import tensorflow as tf
from keras_contrib.losses import DSSIMObjective


def unet_model_3d(input_shape):
    filter1 = 16
    filter2 = filter1 * 2
    filter3 = filter2 * 2
    filter4 = filter3 * 2
    filter5 = filter4 * 2

    inputs = Input(input_shape)
    conv1 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print(pool1.shape)

    conv2 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print(pool2.shape)

    conv3 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print(pool3.shape)

    conv4 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    print(pool4.shape)

    conv5 = Conv3D(filter5, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(filter5, (3, 3, 3), activation='relu', padding='same')(conv5)
    
    up6 = Conv3DTranspose(filter4, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=4)
    conv6 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(conv6)
    print(conv6.shape)

    up7 = Conv3DTranspose(filter3, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=4)
    conv7 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv3DTranspose(filter2, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=4)
    conv8 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv3DTranspose(filter1, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=4)
    conv9 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(conv9)

    residual = Conv3D(1, (1, 1, 1))(conv9)
    outputs = add([inputs, residual])

    model = Model(inputs=inputs, outputs=residual)
    # model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=0.00001), loss='mean_squared_error', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.00001), loss=DSSIMObjective(), metrics=[ssim])

    model.summary()

    return model

def ssim(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    u_true = K.mean(y_true_f)
    u_pred = K.mean(y_pred_f)
    var_true = K.var(y_true_f)
    var_pred = K.var(y_pred_f)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)

    c1 = 0.01
    c2 = 0.03

    return ((2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)) / ((u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2))

# def dssim_loss(y_true, y_pred):
#     ssim_val = ssim(y_true, y_pred)
#     ssim_val = tf.select(tf.is_nan(ssim_val), K.zeros_like(ssim_val), ssim_val)
#     return K.mean(((1.0 - ssim_val) / 2))


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_output_shape(input_shape, depth, filters):
    output_shape = np.divide(input_shape, np.power((2, 2, 2), depth)).tolist()
    return tuple([None] + output_shape + [filters])
