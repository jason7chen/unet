import tensorflow as tf
from unet_model import *
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import time


epochs = 20
batch_size = 1
original_img_size = (256, 256, 96, 1)

config = tf.ConfigProto()
sess = tf.Session(config = config)

# # read data
# x = sio.loadmat('/Users/jasoncjs/Documents/GitHub/unet/train.mat')
# y = sio.loadmat('/Users/jasoncjs/Documents/GitHub/unet/target.mat')
# x = x['train']
# y = y['phantom']

# x = x.reshape((x.shape[0],) + original_img_size)
# y = y.reshape((y.shape[0],) + original_img_size)

# x_train = x[:16,:,:,:,:]
# y_train = y[:16:,:,:,:]
# x_val = x[16:-1,:,:,:,:]
# y_val = y[16:-1,:,:,:,:]
# x_test = x[-1:,:,:,:,:]

# # nromalization
# x_train = x_train.astype('float32')
# x_train -= np.mean(x_train)
# x_train /= np.std(x_train)  

model = unet_model_3d(original_img_size)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
start_time = time.time()
# history = model.fit(x_train, y_train, shuffle = True, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val))
end_time = time.time()
print(end_time - start_time)
# model.save_weights('weights.h5')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.show()
