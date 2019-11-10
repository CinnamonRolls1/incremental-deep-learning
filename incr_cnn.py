import numpy as np 
from full_NN import Neural_Network
from convolution import Convolution 
import scipy.misc as scp
from CNN import CNN
from learning_automata import PowerfulReinforcement

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

	def train(self,X=None,y=None,nodesNumLayer=[16],nodes_outputLayer=1,learning_rate=0.1,epochs=100,hidden_activation=None,output_activation = None):
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
				convoluted_op = self.conv.feed_through_layer(X[j,:])
				cop_shape = convoluted_op.shape
				#print("cop_shape",cop_shape)
				convoluted_op = convoluted_op.flatten()
				y_pred = self.feedForward(convoluted_op,hidden_activation,output_activation)
				cost_sum  += np.sum(cost_function(y_pred,y[j]))
				self.backProp(cost_function,y[j],y_pred,X[j,:],cop_shape)

			
			print("cost:",cost_sum)
			print()



def main():
	model = CNN_Dynamic()

	model.train(X=np.arange(100).reshape(10,10),y=np.arange(10),nodesNumLayer=[32],nodes_outputLayer=10,learning_rate=0.1,epochs = 500,hidden_activation = model.sigmoid, output_activation = model.sigmoid)

if __name__ == '__main__':
	main()
		