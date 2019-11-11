import numpy as np 
import pandas as pd 
import math 
import scipy.misc as scp
import pickle
from sklearn.metrics import accuracy_score

class Neural_Network :

	def __init__(self):
		pass

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
		self.init_weights_bias(X.shape[1])
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		cost_function=self.MSE

		for i in range(epochs) :
			cost_sum = 0
			print("epoch",i)
			for j in range(X.shape[0]) :
				y_pred = self.feedForward(X[j,:],hidden_activation,output_activation)
				cost_sum  += np.sum(cost_function(y_pred,y[j]))
				self.backProp(cost_function,y[j],y_pred)

			
			print("cost:",cost_sum)
			print()


	def init_weights_bias(self,input_shape):
		#print(input_shape)
		self.nodesNumLayer.insert(0,input_shape)
		self.nodesNumLayer.append(self.nodes_outputLayer)

		self.weights = []
		self.bias = []

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
			#print(self.weights[i].shape)


	def feedForward(self,instance,hidden_activation=None, output_activation = None):

		#print()
		hidden_act_vectorized = np.vectorize(self.hidden_activation)
		output_act_vectorized = np.vectorize(self.output_activation)
		current = instance.reshape(1,-1)
		self.layer_vals = [current]

		i=0
		while(i<(len(self.weights)-1)) :
			current = np.matmul(current,self.weights[i])
			current=current + self.bias[i]
			self.layer_vals.append(current)
			current = hidden_act_vectorized(current)
			i+=1

		current = np.matmul(current,self.weights[i]) + self.bias[i]
		current = output_act_vectorized(current)
		self.layer_vals.append(current)
		#print(current)

		#print(current,"\n")

		return current
		


	def backProp(self,cost_function,actual,predicted):
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
		

		for i in range(len(self.weights)-1) :

			x,y = np.meshgrid(errors[i],np.vectorize(self.hidden_activation)(self.layer_vals[-(i+2)]))
			delta_w = x*y*self.lr
			self.weights[-(i+1)] += delta_w

			self.bias[-(i+1)] += (self.lr*errors[i].flatten())


		x,y = np.meshgrid(errors[len(self.weights)-1],self.layer_vals[0])
		delta_w=x*y*self.lr
		self.weights[0] += delta_w
		self.bias[0] += (self.lr*errors[len(self.weights)-1].flatten())


	def test(self,X,y):
		y_pred = np.empty((0,))
		count=0
		for i in range(X.shape[0]) :
			y_pred = self.feedForward(X[i,:],self.hidden_activation,self.output_activation)

			if np.argmax(y_pred) == np.argmax(y[i]) :
				count+=1

		return count/y.shape[0]
			

		#return accuracy_score(y,y_pred)







	def MSE(self,predicted,actual):
		return ((actual-predicted)**2)/2


	def sigmoid(self,x) :
		return (1/(1+math.exp(-x)))

	def reLU(self,x) :
		return (x if x>0 else 0)

	def save(self,filename):
		file = open(filename,"wb")
		pickle.dump(self,file)

	def load(self,filename):
		file = open(filename,"rb")
		return pickle.load(file)



def main():
	model = Neural_Network()
	X_train = np.load('X_train.npy')
	X_train =(X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
	y_train = np.load('y_train.npy')
	X_train = np.asarray([X_train[i,:,:].flatten() for i in range(60000)])
	y = np.empty((0,10))
	for i in range(y_train.shape[0]) :
		row = np.full((10,),0)
		row[y_train[i]] = 1
		row = row.reshape(1,-1)
		y = np.append(y,row,axis =0)

	y_train = y
	#print(X_train.shape,y_train.shape)
	#print(y_train[0])

	#print(model.feedForward())
	#print(X_train[1000,:].shape)
	model.train(X=X_train[:10],y=y_train[:10],nodesNumLayer=[16],nodes_outputLayer=10,learning_rate=0.1,epochs = 10,hidden_activation = model.sigmoid, output_activation = model.sigmoid)
	print(model.test(X_train[:100],y_train[:100]))

if __name__ == '__main__':
	main()
		
		
