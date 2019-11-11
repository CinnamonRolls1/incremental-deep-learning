import numpy as np 
from full_NN import Neural_Network
from convolution import Convolution 
import scipy.misc as scp
from CNN import CNN
from learning_automata import PowerfulReinforcement
import pickle

class CNN_Dynamic(CNN) :

	def __init__(self,num_filt=64):
		CNN.__init__(self,num_filt)

	def init_weights_bias(self,input_shape):
			#print(input_shape)
			self.nodesNumLayer.insert(0,input_shape)
			self.nodesNumLayer.append(self.nodes_outputLayer)

			self.weights = []
			self.bias = []

			self.LA_weights = []

			# print(self.weights[0])
			# print(self.weights[1])
			# print()
			# print(self.bias[0])
			# print(self.bias[1])

			# self.weights.append(np.asarray([[0.2,-0.3],[0.4,0.1],[-0.5,0.2]]))
			# self.weights.append(np.asarray([[-0.3],[-0.2]]))

			# self.bias.append(np.asarray([-0.4,0.2]))
			# self.bias.append(np.asarray([0.1]))
				
			for i in range(len(self.nodesNumLayer)-1) :

				#print(self.nodesNumLayer[i],self.nodesNumLayer[i+1])
				self.weights.append(np.random.normal(0,0.01,(self.nodesNumLayer[i],self.nodesNumLayer[i+1])))
				self.bias.append(np.random.normal(0,0.01,self.nodesNumLayer[i+1]))
				self.LA_weights.append(np.asarray([[PowerfulReinforcement() for k in range(self.nodesNumLayer[i+1])] for j in range(self.nodesNumLayer[i])]))
				print(self.weights[i].shape)
				print(self.LA_weights[i].shape,self.LA_weights[i].dtype)

	def train(self,X=None,y=None,nodesNumLayer=[16],nodes_outputLayer=1,learning_rate=0.1,epochs=100,hidden_activation=None,output_activation = None,save=False):
		if X is None or y is None :
			print('No Dataset or labels given')
			exit()

		if hidden_activation is None or output_activation is None :
			print("Activation function not given")
			exit()


		self.nodesNumLayer=nodesNumLayer #list of number of nodes in each layer
		self.nodes_outputLayer=nodes_outputLayer 
		self.lr = learning_rate		
		self.init_weights_bias(self.conv.feed_through_layer(X[0,:]).flatten().shape[0])
		#self.init_weights_bias(10)
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		cost_function=self.MSE

		for i in range(epochs) :
			cost_sum = 0
			print("epoch",i)
			for j in range(X.shape[0]) :
				self.update_LA()
				#print(list(map(lambda x : x.P,self.LA_weights[0].flatten().tolist())))
				convoluted_op = self.conv.feed_through_layer(X[j,:])
				cop_shape = convoluted_op.shape
				#print("cop_shape",cop_shape)
				convoluted_op = convoluted_op.flatten()
				y_pred = self.feedForward(convoluted_op,hidden_activation,output_activation)
				cost_sum  += np.sum(cost_function(y_pred,y[j]))
				self.backProp(cost_function,y[j],y_pred,X[j,:],cop_shape)

			
			print("cost:",cost_sum)
			print()

		if save :
			self.save("incr_cnn.obj")

def retrain(self,X=None,y=None,edge_learning_rate=0.1,core_learning_rate=0.001,epochs=100,save=False):
		if X is None or y is None :
			print('No Dataset or labels given')
			exit()

		if hidden_activation is None or output_activation is None :
			print("Activation function not given")
			exit()


		#self.nodesNumLayer=nodesNumLayer #list of number of nodes in each layer
		#self.nodes_outputLayer=nodes_outputLayer 
		self.core_lr = core_learning_rate
		self.edge_lr = edge_learning_rate
		#self.init_weights_bias(self.conv.feed_through_layer(X[0,:]).flatten().shape[0])
		#self.init_weights_bias(10)
		#self.hidden_activation = hidden_activation
		#self.output_activation = output_activation
		cost_function=self.MSE

		for i in range(epochs) :
			cost_sum = 0
			print("epoch",i)
			for j in range(X.shape[0]) :
				self.update_LA()
				#print(list(map(lambda x : x.P,self.LA_weights[0].flatten().tolist())))
				convoluted_op = self.conv.feed_through_layer(X[j,:])
				cop_shape = convoluted_op.shape
				#print("cop_shape",cop_shape)
				convoluted_op = convoluted_op.flatten()
				y_pred = self.feedForward(convoluted_op,hidden_activation,output_activation)
				cost_sum  += np.sum(cost_function(y_pred,y[j]))
				self.dynamic_backProp(cost_function,y[j],y_pred,X[j,:],cop_shape)

			
			print("cost:",cost_sum)
			print()

		if save :
			self.save("incr_cnn.obj")

	def dynamic_backProp(self,cost_function,actual,predicted,inp,conv_layer_shape):
		errors = []
		derivative_cost = scp.derivative(self.MSE,actual,dx=1e-6,args=(predicted,))
		output_layer = self.layer_vals[-1]
		output_error = derivative_cost*np.vectorize(scp.derivative)(self.output_activation,output_layer,dx=1e-6)
		current_err = output_error.reshape(1,-1)
		i=1
		while i<=(len(self.weights)) :
			errors.append(current_err)
			current_layer = self.layer_vals[-(i+1)]
			current_err = np.matmul(current_err,self.weights[-i].T)*np.vectorize(scp.derivative)(self.hidden_activation,current_layer,dx=1e-6)
			i+=1

		self.conv.conv_backProp(current_err.reshape(conv_layer_shape[0],conv_layer_shape[1],conv_layer_shape[2]),inp)

		for i in range(len(self.weights)-1) :

			x,y = np.meshgrid(errors[i],np.vectorize(self.hidden_activation)(self.layer_vals[-(i+2)]))
			delta_w = x*y

			lrs = np.where(self.LA_weights[-(i+1)].get_alpha() == 1,self.core_lr,self.edge_lr)
			delta_w*=lrs

			self.weights[-(i+1)] += delta_w

			self.bias[-(i+1)] += (self.lr*errors[i].flatten())


		x,y = np.meshgrid(errors[len(self.weights)-1],self.layer_vals[0])
		delta_w=x*y
		lrs = np.where(self.LA_weights[0].get_alpha() == 1,self.core_lr,self.edge_lr)
		delta_w*=lrs
		self.weights[0] += delta_w
		self.bias[0] += (self.lr*errors[len(self.weights)-1].flatten())


	def update_LA(self):
		vect_update_LA = np.vectorize(self.update_LA_elem)
		for i in range(len(self.weights)) :
			vect_update_LA(self.weights[i],self.LA_weights[i])

	def update_LA_elem(self,weight,la):
		la.learn(weight)




def main():
	model = CNN_Dynamic(4)

	X_train = np.load('X_train.npy')
	X_train =(X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
	y_train = np.load('y_train.npy')
	#X_train = np.asarray([X_train[i,:,:].flatten() for i in range(60000)])
	y = np.empty((0,10))
	for i in range(y_train.shape[0]) :
		row = np.full((10,),0)
		row[y_train[i]] = 1
		row = row.reshape(1,-1)
		y = np.append(y,row,axis =0)

	y_train = y

	model.train(X=X_train[:1000,:],y=y_train[:1000],nodesNumLayer=[16],nodes_outputLayer=10,learning_rate=0.1,epochs = 500,hidden_activation = model.sigmoid, output_activation = model.sigmoid,save=True)
	print(model.weights[0][0][0])

	new_model=CNN_Dynamic().load("incr_cnn.obj")
	print(new_model.weights[0][0][0])

if __name__ == '__main__':
	main()
		