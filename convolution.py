from PIL import Image
import numpy as np

class Convolution : #BUilds a convolution layer

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

		for i in range(0,filtered_img.shape[0]-pool_dim+1,stride) :

			temp = np.empty(shape=0)
			for j in range(0,filtered_img.shape[1]-pool_dim+1,stride) :

				max_val = np.max(filtered_img[i:i+pool_dim,j:j+pool_dim])

				temp = np.append(temp,max_val)

			#print(temp.shape,filtered_img.shape)
			#print(temp.shape)
			pooled_img =  np.append(pooled_img,temp.reshape(1,-1),axis=0)

		return pooled_img #returns a numpy array of shape with similar dimennsional reduction as filtering(refer line 30 and 31)


	def apply_maxPool_to_filters(self,filtered_img) : #applies pooling to all the filtered versions of an image

		pooled_layer = []

		for i in range(filtered_img.shape[0]) :

			pooled_layer.append(self.apply_maxPool(filtered_img))

		return np.asarray(pooled_layer)	#returns a numpy array with shape(<num_filters>,x-axis shape of pooled image, y-axis shape of pooled image)



	def reLU(x) :
		return (x if x>0 else 0)

	def apply_activation(self,inp, act_func=reLU) : #applies activation function on the filter image
		
		reLU_vect = np.vectorize(act_func)

		return reLU_vect(inp) #return numpy array same shape as <inp>


	def feed_through_layer(self,inp): #takes an image fllters it first, then applies an activation and finally performs pooling
		return self.apply_maxPool_to_filters(self.apply_activation(self.apply_filters(inp,self.filter_bank))) #returns a numpy array of shape (<num_filters>,x-axis shape of pooled image, y-axis shape of pooled image)



def rgb2gray(rgb) :
	r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
	gray = 0.2989*r + 0.5870*g + 0.1140*b
	return gray


def main():
	img = Image.open('img.jpeg') #sample image
	img = np.asarray(img)
	img = rgb2gray(img)#convert to grayscale
	print("Shape of image: ",img.shape)

			
	'''img = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

	filter = np.asarray([[3,3,3],[3,3,3],[3,3,3]])'''

	#edge_detect =convolve(img,filter)

	#plt.imshow(img)
	#plt.show()

	'''plt.imshow(edge_detect)
	plt.show()'''
	#print(apply_activation([[1,2,3,4,5],[-1,-2,-3,-4,-5]]))

	conv = Convolution()
	print("Number of filters: ",conv.filter_bank.shape[0])
	print("Filter shape: ","(",conv.filter_bank.shape[1],",",conv.filter_bank.shape[2],")")
	print("Stride: ",1)

	print("Convolution layer shape: ",conv.feed_through_layer(img).shape)


if __name__ == '__main__':# run main to run the convolution function on a single image
	main()