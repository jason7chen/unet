import numpy as np
from keras.engine import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, add
from keras.optimizers import Adam
from keras_contrib.layers.convolutional import Deconvolution3D
from keras import backend as K


def unet_model_3d(input_shape):
    filter1 = 32
    filter2 = filter1 * 2
    filter3 = filter2 * 2
    filter4 = filter3 * 2
    filter5 = filter4 * 2

    inputs = Input(input_shape)
    conv1 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(filter5, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(filter5, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = Deconvolution3D(filter4, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool4.shape, output_shape=get_output_shape(input_shape[:3], 3, filter4))(conv5)
    up6 = concatenate([up6, conv4], axis=4)
    conv6 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(filter4, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Deconvolution3D(filter3, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool3.shape, output_shape=get_output_shape(input_shape[:3], 2, filter3))(conv6)
    up7 = concatenate([up7, conv3], axis=4)
    conv7 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(filter3, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Deconvolution3D(filter2, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool2.shape, output_shape=get_output_shape(input_shape[:3], 1, filter2))(conv7)
    up8 = concatenate([up8, conv2], axis=4)
    conv8 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(filter2, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = Deconvolution3D(filter1, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=pool1.shape, output_shape=get_output_shape(input_shape[:3], 0, filter1))(conv8)
    up9 = concatenate([up9, conv1], axis=4)
    conv9 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(filter1, (3, 3, 3), activation='relu', padding='same')(conv9)

    residual = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)
    outputs = add([inputs, residual])

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

def get_output_shape(input_shape, depth, filters):
    output_shape = np.divide(input_shape, np.power((2, 2, 2), depth)).tolist()
    return tuple([None] + output_shape + [filters])
