import numpy as np 
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() 


def init_filters(num_filters,filter_shape) :

	return np.random.normal(size = (num_filters,filter_shape[0],filter_shape[1]))

num_filters = 64
filter_shape = (3,3)
stride = 1

filter_bank = init_filters(num_filters,filter_shape)

print(filter_bank.shape)