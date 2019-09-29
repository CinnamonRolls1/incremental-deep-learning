import numpy as np 
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

np.save("X_train",X_train)
np.save("y_train",y_train)
np.save("X_test",X_test)
np.save("y_test",y_test)

