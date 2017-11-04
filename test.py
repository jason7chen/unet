import tensorflow as tf
from unet_model import *
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import time


epochs = 50
batch_size = 1
original_img_size = (64, 64, 32, 1)

config = tf.ConfigProto()
sess = tf.Session(config = config)

# read data
x = sio.loadmat('/home/ubuntu/unet/train.mat')
y = sio.loadmat('/home/ubuntu/unet/target.mat')
x = x['train']
y = y['phantom']

x = x.reshape((x.shape[0],) + original_img_size)
y = y.reshape((y.shape[0],) + original_img_size)

x_test = x[-1:,:,:,:,:]

# # nromalization
# x_train = x_train.astype('float32')
# x_train -= np.mean(x_train)
# x_train /= np.std(x_train)  

model = unet_model_3d(original_img_size)
print(x_test.shape)
model.load_weights('weights.h5')
start_time = time.time()
y_test = model.predict(x_test, batch_size=batch_size)
end_time = time.time()
print(end_time - start_time)

sio.savemat('test.mat', {'test':y_test})
