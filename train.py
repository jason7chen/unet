import tensorflow as tf
from unet_model import *
import scipy.io as sio 
import numpy as np


epochs = 50
batch_size = 1
original_dim = 250*250
original_img_size = (256, 256, 128, 1)

config = tf.ConfigProto()
sess = tf.Session(config = config)

# read data
x = sio.loadmat('/Users/jasoncjs/Documents/MATLAB/QSM/train.mat')
y = sio.loadmat('/Users/jasoncjs/Documents/MATLAB/QSM/target.mat')
x = x['phase']
y = y['phantom']

x = x.reshape((x.shape[0],) + original_img_size)
y = y.reshape((y.shape[0],) + original_img_size)

x_train = x[:4,:,:,:,:]
y_train = y[:4:,:,:,:]
x_val = x[4:,:,:,:,:]
y_val = y[4:,:,:,:,:]

# # nromalization
# x_train = x_train.astype('float32')
# x_train -= np.mean(x_train)
# x_train /= np.std(x_train)  

model = unet_model_3d(original_img_size)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
history = model.fit(x_train, y_train, shuffle = True, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

