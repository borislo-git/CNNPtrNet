import numpy as np
from .BaseProblem import BaseProblem

class SortingProblem(BaseProblem):

	def makeBatch(self, params):
		return np.random.rand(params['batchSize'], params['length'], 1)

	def getSolution(self, batchArray):
		return np.sort(batchArray, axis=1)

	def getArgs(self, batchArray, solutionArray=None):
		return np.argsort(batchArray, axis=1)

	def accuracy(self, exactSolution, solution, batchArray):
		return len([1 for batch in range(exactSolution.shape[0]) 
			if sum(exactSolution[batch,1:,0]==batchArray[batch,:,:].T[0][solution[batch,:]]) == solution.shape[1]])/exactSolution.shape[0]



