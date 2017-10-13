import numpy as np
from keras.engine import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate
from keras.optimizers import Adam
from keras_contrib.layers.convolutional import Deconvolution3D



def unet_model_3d(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = Deconvolution3D(512, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool4.shape, output_shape=tuple([None] + list(conv4.shape)))(conv5)
    up6 = concatenate([up6, conv4], axis=4)
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Deconvolution3D(256, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool3.shape, output_shape=tuple([None] + list(conv3.shape)))(conv6)
    up7 = concatenate([up7, conv3], axis=4)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Deconvolution3D(128, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool2.shape, utput_shape=tuple([None] + list(conv2.shape)))(conv7)
    up8 = concatenate([up8, conv2], axis=4)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = Deconvolution3D(64, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool1.shape, output_shape=tuple([None] + list(conv1.shape)))(conv8)
    up9 = concatenate([up9, conv1], axis=4)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv3D((1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()

    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

