from PIL import Image
import numpy as np
import scipy.misc as scp

def reLU(x) :
	return (x if x>0 else 0)


class Convolution : #Builds a convolution layer

	def __init__(self,num_filters=64,filter_shape=(3,3)):
		self.filter_bank = self.init_filters(num_filters,filter_shape)
		

	def init_filters(self,num_filters,filter_shape) :# initialize <num_filters> number of filter with <filter_shape> shape

		return np.random.normal(size = (num_filters,filter_shape[0],filter_shape[1]))  #returns numpy array of shape (<num_filters>,<filter_shape[0]>,<filter_shape[1]>)


	def convolve(self,img,filter,stride=1) : # applies a on the image
		if len(img.shape) != len(filter.shape) :
			print('filter depth should match image depth')
			exit()

		new_col_len = (img.shape[1]-filter.shape[1]+1)//stride
		filtered_img = np.empty(shape=[0,new_col_len])
		for i in range(0,img.shape[0]-filter.shape[0]+1,stride) :

			temp = np.empty(shape=0)
			for j in range(0,img.shape[1]-filter.shape[1]+1,stride) :

				masked_values = np.sum(np.multiply(img[i:i+filter.shape[0],j:j+filter.shape[1]], filter))

				temp = np.append(temp,masked_values) 

			#print(temp.shape,filtered_img.shape)
			filtered_img =  np.append(filtered_img,temp.reshape(1,-1),axis=0)

		return filtered_img #returns numpy array of shape (net rows after applying filter, net columns after applying filter)
		#					e.g. img->(4,4) filter->(3,3) stride=1 after applyinh filter shape->((4-3+1)/stride, (4-3+1)//stride) -> (2,2)

	def apply_filters(self,img, filter_bank) : #convolves an image with all the filters in the <filter_bank>

		filtered_layer = []

		for i in filter_bank :

			filtered_layer.append(self.convolve(img,i))

		return np.asarray(filtered_layer) #returns a numpy array of shape (<num_filters>,filtered image x-axis shape,filtered image y-axis shape)



	def apply_maxPool(self,filtered_img,stride=2,pool_dim=2) : #performs pooling on the filters image

		new_col_len = ((filtered_img.shape[1]-pool_dim)//stride )+ 1
		#print("new_col_len",new_col_len)
		pooled_img = np.empty(shape=[0,new_col_len])

		pool_deriv = []
		for i in range(0,filtered_img.shape[0]-pool_dim+1,stride) :

			temp = np.empty(shape=0)
			for j in range(0,filtered_img.shape[1]-pool_dim+1,stride) :

				max_val = np.max(filtered_img[i:i+pool_dim,j:j+pool_dim])
				flat_index = np.argmax(filtered_img[i:i+pool_dim,j:j+pool_dim])
				index = (flat_index//pool_dim,flat_index%pool_dim)
				index = (index[0]+i,index[1]+j)
				pool_deriv.append(index)

				temp = np.append(temp,max_val)

			#print(temp.shape,filtered_img.shape)
			#print(temp.shape)
			pooled_img =  np.append(pooled_img,temp.reshape(1,-1),axis=0)

		return pooled_img,pool_deriv #returns a numpy array of shape with similar dimennsional reduction as filtering(refer line 30 and 31)


	def apply_maxPool_to_filters(self,filtered_img) : #applies pooling to all the filtered versions of an image

		pooled_layer = []
		pooled_deriv_layer = []

		for i in range(filtered_img.shape[0]) :

			temp = self.apply_maxPool(filtered_img[i])
			pooled_layer.append(temp[0])
			pooled_deriv_layer.append(temp[1])

		return np.asarray(pooled_layer),pooled_deriv_layer	#returns a numpy array with shape(<num_filters>,x-axis shape of pooled image, y-axis shape of pooled image)


	def maxPool_backProp(self,dh,cached_input,prev_layer_shape):
		d_pool = np.zeros(prev_layer_shape)

		for i in range(len(cached_input)) :
			d_pool[cached_input[i][0],cached_input[i][1]]=dh[i]

		return d_pool

	def activation_backProp(self,dh,cached_input) :
		return dh*np.vectorize(scp.derivative)(reLU,cached_input,dx=1e-6)

	def convolve_backProp(self,	dh,cached_input,filter,stride=1):
		
		new_col_len = (cached_input.shape[1]-dh.shape[1]+1)//stride

		dw = np.empty(shape=[0,new_col_len])
		for i in range(0,cached_input.shape[0]-dh.shape[0]+1,stride) :

			temp = np.empty(shape=0)
			for j in range(0,cached_input.shape[1]-dh.shape[1]+1,stride) :

				masked_values = np.sum(np.multiply(cached_input[i:i+dh.shape[0],j:j+dh.shape[1]], dh))

				temp = np.append(temp,masked_values) 

			#print(temp.shape,dw.shape)
			dw =  np.append(dw,temp.reshape(1,-1),axis=0)

		return dw

	def MSE(self,predicted,actual):
		return ((actual-predicted)**2)/2

	def apply_activation(self,inp, act_func=reLU) : #applies activation function on the filter image
		
		reLU_vect = np.vectorize(act_func)

		return reLU_vect(inp) #return numpy array same shape as <inp>


	def feed_through_layer(self,inp): #takes an image fllters it first, then applies an activation and finally performs pooling
		filtered_imgs = self.apply_filters(inp,self.filter_bank)
		self.activated_layer = self.apply_activation(filtered_imgs)
		pooled_layer,self.pooled_deriv = self.apply_maxPool_to_filters(self.activated_layer)
		#print(pooled_layer)





		return pooled_layer

	def conv_backProp(self,dh,inp):

		for i in range(self.filter_bank.shape[0]) :
			d_pool = self.maxPool_backProp(dh[i].reshape(-1,),self.pooled_deriv[i],self.activated_layer[0].shape)
			deriv_activ = self.activation_backProp(d_pool,self.activated_layer[i])
			dw = self.convolve_backProp(deriv_activ,inp,self.filter_bank[i])
			self.filter_bank[i] += dw

def rgb2gray(rgb) :
	r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
	gray = 0.2989*r + 0.5870*g + 0.1140*b
	return gray


def main():
	img = Image.open('img.jpeg') #sample image
	img = np.asarray(img)
	img = rgb2gray(img)#convert to grayscale
	print("Shape of image: ",img.shape)

			
	#img = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

	#filter = np.asarray([[3,3,3],[3,3,3],[3,3,3]])

	#edge_detect =convolve(img,filter)

	#plt.imshow(img)
	#plt.show()

	'''plt.imshow(edge_detect)
	plt.show()'''
	#print(apply_activation([[1,2,3,4,5],[-1,-2,-3,-4,-5]]))

	conv = Convolution(2)
	print("Number of filters: ",conv.filter_bank.shape[0])
	print("Filter shape: ","(",conv.filter_bank.shape[1],",",conv.filter_bank.shape[2],")")
	print("Stride: ",1)

	print("Convolution layer : \n",conv.feed_through_layer(img).shape)


if __name__ == '__main__' : 
	main()