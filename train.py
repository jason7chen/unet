import tensorflow as tf
from unet_model import *


epochs = 50
# batch_size = 
original_dim = 250*250
original_img_size = (256, 256, 128, 1)

config = tf.ConfigProto()
sess = tf.Session(config=config)

# # read data
# x_train = 
# y_train = 

# # nromalization
# x_train = x_train.astype('float32')
# x_train -= np.mean(x_train)
# x_train /= np.std(x_train)  

model = unet_model_3d(original_img_size)