import numpy as np 
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist

from PIL import Image
#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() 


def init_filters(num_filters,filter_shape) :

	return np.random.normal(size = (num_filters,filter_shape[0],filter_shape[1]))

num_filters = 64
filter_shape = (3,3)
stride = 1

filter_bank = init_filters(num_filters,filter_shape)

def convolve(img,filter,stride=1) :
	if len(img.shape) != len(filter.shape) :
		raise Exception('filter depth should match image depth')

	new_col_len = (img.shape[1]-filter.shape[1]+1)//stride
	filtered_img = np.empty(shape=[0,new_col_len])
	for i in range(0,img.shape[0]-filter.shape[0]+1,stride) :

		temp = np.empty(shape=0)
		for j in range(0,img.shape[1]-filter.shape[1]+1,stride) :

			masked_values = np.sum(np.multiply(img[i:i+filter.shape[0],j:j+filter.shape[1]], filter))

			temp = np.append(temp,masked_values)

		#print(temp.shape,filtered_img.shape)
		filtered_img =  np.append(filtered_img,temp.reshape(1,-1),axis=0)

	return filtered_img

'''img = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

filter = np.asarray([[3,3,3],[3,3,3],[3,3,3]])'''

def rgb2gray(rgb) :
	r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
	gray = 0.2989*r + 0.5870*g + 0.1140*b

	return gray

img = Image.open('img.jpeg')
img = np.asarray(img)
print(img.shape)
filter = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

img = rgb2gray(img)
#edge_detect =convolve(img,filter)

#plt.imshow(img)
#plt.show()

'''plt.imshow(edge_detect)
plt.show()'''

def apply_filters(img, filter_bank) :

	filtered_layer = []

	for i in filter_bank :

		filtered_layer.append(convolve(img,i))

	return np.asarray(filtered_layer)


#print(img_to_convolution_layer(img,filter_bank).shape)



def apply_maxPool(filtered_img,stride=2,pool_dim=2) :

	new_col_len = ((filtered_img.shape[1]-pool_dim)//stride )+ 1
	#print("new_col_len",new_col_len)
	pooled_img = np.empty(shape=[0,new_col_len])

	for i in range(0,filtered_img.shape[0]-pool_dim+1,stride) :

		temp = np.empty(shape=0)
		for j in range(0,filtered_img.shape[1]-pool_dim+1,stride) :

			max_val = np.max(filtered_img[i:i+pool_dim,j:j+pool_dim])

			temp = np.append(temp,max_val)

		#print(temp.shape,filtered_img.shape)
		#print(temp.shape)
		pooled_img =  np.append(pooled_img,temp.reshape(1,-1),axis=0)

	return pooled_img

#print(apply_maxPool(np.arange(16).reshape(4,4)))

def apply_maxPool_to_filters(filtered_img) :

	pooled_layer = []

	for i in range(filtered_img.shape[0]) :

		pooled_layer.append(apply_maxPool(img))

	return np.asarray(pooled_layer)	


filtered_imgs = apply_filters(img,filter_bank)
print(filtered_imgs.shape)
pooled_imgs = apply_maxPool_to_filters(filtered_imgs)
print(pooled_imgs.shape)

def reLU(x) :
	return (x if x>0 else 0)

def apply_activation(inp, act_func=reLU) :
	pass