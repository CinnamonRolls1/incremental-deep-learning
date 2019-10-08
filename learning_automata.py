import numpy as np

class PowerfulReinforcement :
	def __init__(self):
		self.alpha= [0,1] #activate-1 deactivate-0 
		self.P = [0.5,0.5]
		self.beta = [0,1]
		self.threshold = 0 
		self.step = 0.1


	def learn(self,weight):
		
		alpha_i = self.get_alpha()

		feedback = self.get_feedback(weight,alpha_i)

		if feedback ==1:
			self.update(alpha_i)


	def get_alpha(self):
		return self.alpha[0] if self.P[0] >= self.P[1] else self.alpha[1]

	def get_feedback(self,weight,alpha_i):
		
		if alpha_i == 1 :
			return 1 if weight > self.threshold else 0

		else :
			return 1 if weight < self.threshold else 0

	def update(self,alpha_i):

		if alpha_i == 0:
			self.P[1] = max(self.P[1]-self.step,0)
			self.P[0] = 1-self.P[1]

		else:
			self.P[0] = max(self.P[0]-self.step,0)
			self.P[1] = 1-self.P[0]


def main() :
	automata = PowerfulReinforcement()

	i=-5
	while(i<=5) :
		automata.learn(i)
		print(i,automata.P)
		i+=0.01



if __name__ == '__main__':
	main()

		

