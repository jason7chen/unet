import numpy as np
from keras.engine import Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.optimizers import Adam
from keras_contrib.layers import Deconvolution3D



def unet_model_3d():
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

    conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = Deconvolution3D(512, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=compute_shape(512, 4, (2, 2, 2), input_shape), 
    																	 output_shape=compute_shape(512, 3, (2, 2, 2), input_shape))
    up6 = concatenate([up6, conv4], axis=)
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Deconvolution3D(256, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=, output_shape=)
    up7 = concatenate([up7, conv3], axis=)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Deconvolution3D(128, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=, output_shape=)
    up8 = concatenate([up8, conv2], axis=)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = Deconvolution3D(64, kernel_size=(2, 2, 2), strides=(2, 2, 2), input_shape=, output_shape=)
    up9 = concatenate([up9, conv1], axis=)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D((1, 1, 1))(conv9)
    outputs = Activation('sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def compute_shape(filters, depth, pool_size, image_shape):
	if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + output_image_shape)
