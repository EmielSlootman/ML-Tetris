import numpy as np

class MSE_loss:
	def get_loss(self, H, Y):
		m = len(Y)
		L = 1/(2*m) * np.sum((H - Y)**2)
		return L
	
	def diff(self,H,Y):
		m = len(Y)
		dL_dZ = 1/m * H-Y
		return dL_dZ

class CE_loss:
	def get_loss(self, H, Y):
		L = np.sum(np.dot(-Y.T , np.log(H)))/Y.shape[1]
		return L
	
	def diff(self,H,Y):
		n = Y.shape[0]
		dL_dZ = 1/n*(H-Y)
		return dL_dZ