import numpy as np
import struct
import matplotlib
import matplotlib.pyplot as plt
import collections
import random
import math
import tetris
import activations as act

def init_theta(n1,n2,activation):
	if activation in [act.sigmoid,act.softmax]:
		M = np.random.randn(n2,n1)*np.sqrt(2./n1)
	elif activation in [act.relu] :
		M = np.random.randn(n2,n1)*np.sqrt(1./n1)
	elif activation == act.tanh:
		M = np.random.randn(n2,n1)*np.sqrt(1./(n1+n2))
	else:
		M = np.random.randn(n2,n1)
	return M

class layer:
	def __init__(self, n_prev, n_next, activation):
		self.W = init_theta(n_prev, n_next, activation)
		self.B = init_theta(1, n_next, activation)
		self.activation = activation()
		
	def forward(self, A_prev):
		self.Z = np.dot(self.W, A_prev) + self.B
		self.A = self.activation.activate(self.Z)
		return self.A
	
	def grad(self, dL_dZ_next, W_next, A_prev, m):  
		dL_dA = np.dot(W_next.T,dL_dZ_next)
		dA_dZ = self.activation.diff(self.Z)
		self.dZ = dL_dA * dA_dZ
		self.dW = (1./m)*(np.dot(self.dZ, A_prev.T))
		self.dB = (1./m)*(np.sum(self.dZ, axis=1, keepdims=True))
	
	def out_grad(self, dL_dA, A_prev, m):
		dA_dZ = self.activation.diff(self.Z)
		self.dZ = dL_dA * dA_dZ
		self.dW = (1./m)*(np.dot(self.dZ, A_prev.T))
		self.dB = (1./m)*(np.sum(self.dZ))#, axis=1, keepdims=True))
		
	def step(self, lr):
		self.W = self.W - lr * self.dW
		self.B = self.B - lr * self.dB

def SGD(batch_size,X,Y,model,lr=0.001):
	m = len(Y)
	
	model.f_pass(X)
	model.back_prop(X, Y, m)
	model.optim(lr)
	
	return model.loss

class DQN:
	def __init__(self, X_size, Y_size, lossfn):
		self.L1 = layer(X_size, 16, act.relu)
		self.L2 = layer(16, 16, act.relu)        
		self.L3 = layer(16, Y_size, act.linear)
		self.lossfn = lossfn()
		
	def f_pass(self, X):
		A1 = self.L1.forward(X)
		A2 = self.L2.forward(A1)
		A3 = self.L3.forward(A2)
		self.H = A3
		return self.H
	
	def back_prop(self, X, Y, batch_size):
		m = batch_size
		self.loss = self.lossfn.get_loss(self.H, Y)
		dL_dZ = self.lossfn.diff(self.H, Y)

		self.L3.out_grad(dL_dZ, self.L2.A, m)
		self.L2.grad(self.L3.dZ, self.L3.W, self.L1.A, m)
		self.L1.grad(self.L2.dZ, self.L2.W, X, m)
		#print(self.L1.dW, '\n', self.L2.dW, '\n', self.loss, '\n', dL_dZ, '\n')
		
	def optim(self, lr):
		self.L1.step(lr)
		self.L2.step(lr)
		self.L3.step(lr)

class DQNsimple:
	def __init__(self, X_size, Y_size, lossfn):
		self.L1 = layer(X_size, 8, act.relu)     
		self.L2 = layer(8, Y_size, act.linear)
		self.lossfn = lossfn()
		
	def f_pass(self, X):
		A1 = self.L1.forward(X)
		A2 = self.L2.forward(A1)
		self.H = A2
		return self.H
	
	def back_prop(self,X,Y, batch_size):
		m = batch_size
		self.loss = self.lossfn.get_loss(self.H, Y)
		dL_dZ = self.lossfn.diff(self.H, Y)
		self.L2.out_grad(dL_dZ, self.L1.A, m)
		self.L1.grad(self.L2.dZ, self.L2.W, X, m)
		#print(self.L1.dW, '\n', self.L2.dW, '\n', self.loss, '\n', dL_dZ, '\n')
		
	def optim(self, lr):
		self.L1.step(lr)
		self.L2.step(lr)

Experience = collections.namedtuple('Experience', ('state', 'done', 'next_state', 'reward'))

class ReplayMemory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.push_count = 0
	
	def push(self, experience):
		if len(self.memory) < self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.push_count % self.capacity] = experience
		self.push_count += 1

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_exploration_rate(self, current_step):
		return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

