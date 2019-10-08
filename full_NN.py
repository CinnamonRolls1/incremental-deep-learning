import numpy as np 
import pandas as pd 
import math 
import scipy.misc as scp

class Neural_Network :

	def __init__(self):
		pass

	def train(self,X=None,y=None,nodesNumLayer=[16],nodes_outputLayer=1,learning_rate=0.1,epochs=100):
		if X is None or y is None :
			print('No Dataset or labels given')
			exit()

		self.nodesNumLayer=nodesNumLayer #list of number of nodes in each layer
		self.nodes_outputLayer=nodes_outputLayer 
		self.lr = learning_rate		
		self.init_weights_bias(X)
		cost_function=self.MSE

		for i in range(epochs) :
			for j in range(X.shape[0]) :
				y_pred = self.feedForward(X[j,:],self.sigmoid)
				cost  = cost_function(y_pred,y[j])
				self.backProp(cost_function,y[j],y_pred)



	def init_weights_bias(self,X):
		
		self.nodesNumLayer.insert(0,X.shape[1])
		self.nodesNumLayer.append(self.nodes_outputLayer)

		self.weights = [np.asarray([[0.2,-0.3],[0.4,0.1],[-0.5,0.2]]),np.asarray([[-0.3],[-0.2]])]
		self.bias = [np.asarray([-0.4,0.2]),np.asarray([0.1])]

		print(self.weights[0])
		print(self.weights[1])
		print()
		print(self.bias[0])
		print(self.bias[1])


		# for i in range(len(self.nodesNumLayer)-1) :

		# 	self.weights.append(np.random.randn(self.nodesNumLayer[i],self.nodesNumLayer[i+1]))
		# 	self.bias.append(np.random.rand(self.nodesNumLayer[i+1]))
		# 	print(self.weights[i].shape)

	def feedForward(self,instance,activation=None):
		if activation==None :
			print('Activation function not given')
			exit()

		#print()
		act_vectorized = np.vectorize(activation)
		current = instance.reshape(1,-1)
		self.layer_vals = [current]
		for i in range(len(self.weights)) :
			#print(current)
			current = np.matmul(current,self.weights[i]) + self.bias[i]
			self.layer_vals.append(current)
			#print(current,'\n')
			current = act_vectorized(current)

		#print(current,"\n")

		return current
		




	# def backProp(self,cost_function,actual,predicted):
		
	# 	derivative_cost = scp.derivative(cost_function,actual,dx=1e-6,args=(predicted,))
	# 	output_layer = self.layer_vals.pop()
	# 	output_error = derivative_cost*scp.derivative(self.sigmoid,output_layer,dx=1e-6)
	# 	current_err = output_error.reshape(1,-1)
	# 	current_layer = output_layer
	# 	print('\ncurrent_err: ',current_err)
	# 	i=1
	# 	while self.layer_vals != [] :
	# 		current_layer = self.layer_vals.pop()
	# 		print(current_layer.shape,current_err.shape)
	# 		delta_weights = self.lr*9*current_err*np.vectorize(self.sigmoid)(current_layer)
	# 		print("delta_weights",delta_weights)
	# 		current_err = np.matmul(current_err,self.weights[-i].T)
	# 		current_err = current_err*np.vectorize(scp.derivative)(self.sigmoid,current_layer,dx=1e-6)
	# 		self.weights[-i] = self.weights[-i]+delta_weights 
	# 		i+=1

	def backProp(self,cost_function,actual,predicted):
		errors = []
		derivative_cost = scp.derivative(self.MSE,1,dx=1e-6,args=(0.474,))
		output_layer = self.layer_vals[-1]
		output_error = derivative_cost*scp.derivative(self.sigmoid,output_layer,dx=1e-6)
		current_err = output_error.reshape(1,-1)
		i=1
		while i<=len(self.weights) :
			errors.append(current_err)
			current_layer = self.layer_vals[-(i+1)]
			current_err = np.matmul(current_err,self.weights[-i].T)*np.vectorize(scp.derivative)(self.sigmoid,current_layer,dx=1e-6)
			i+=1


		#print(errors,'\n\n')

		for i in range(len(self.weights)-1) :
			#print(errors[i])
			#print(self.weights[-(i+1)])
			x,y = np.meshgrid(errors[i],np.vectorize(self.sigmoid)(self.layer_vals[-(i+2)]))
			delta_w = x*y*self.lr
			self.weights[-(i+1)] += delta_w
			#print(delta_w,'\n')


		x,y = np.meshgrid(errors[len(self.weights)-1],self.layer_vals[0])
		delta_w=x*y*self.lr
		self.weights[0] += delta_w
		#print(delta_w)
		#self.weights[-(i+1)] += delta_w.T



	def MSE(self,predicted,actual):
		return ((actual-predicted)**2)/2

	def sigmoid(self,x) :
		return (1/(1+math.exp(-x)))



def main():
	model = Neural_Network()
	model.train(X=np.asarray([[1,0,1]]),y=np.asarray([1]),nodesNumLayer=[2],nodes_outputLayer=1,learning_rate=0.9)
	print(model.feedForward(np.asarray([1,0,1]),activation = model.sigmoid))

if __name__ == '__main__':
	main()
		
		
