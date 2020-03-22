import numpy as np

class sigmoid:
	def activate(self,Z):
		A = 1/(1+np.exp(-Z))
		return A
	
	def diff(self,Z):
		dA_dZ = np.multiply(self.activate(Z),(1-self.activate(Z)))
		return dA_dZ

class linear:
	def activate(self, Z):
		return Z

	def diff(self, Z):
		return 1
		
class relu:
	def activate(self,Z):
		A = np.maximum(0, Z)
		return A
	
	def diff(self,Z):
		dA_dZ = 1*(Z>0)
		return dA_dZ

class tanh:
	def activate(self,Z):
		A = np.tanh(Z)
		return A

	def diff(self,Z):
		dA_dZ = 1 - (np.multiply(self.activate(Z),self.activate(Z)))
		return dA_dZ
	
class softmax:
	def activate(self,Z):
		e_Z = np.exp(Z- np.max(Z,axis=0))
		A = e_Z / e_Z.sum(axis=0)
		return A

	def diff(self,Z):
		sftmx = self.activate(Z)
		a = np.einsum('ij,jk->ijk',np.eye(sftmx.shape[0]),sftmx)
		b = np.einsum('ij,kj->ikj',sftmx,sftmx)
		dH_dZ = a - b
		return dH_dZ