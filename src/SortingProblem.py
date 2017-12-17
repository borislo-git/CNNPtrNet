import numpy as np
from .BaseProblem import BaseProblem

class SortingProblem(BaseProblem):

	def makeBatch(self, params):
		return np.random.randint(low  = params['low'],
								 high = params['high']+1,
								 size = [params['batchSize'], params['length'], 1])

	def getSolution(self, batchArray):
		return np.sort(batchArray, axis=1)

	def getArgs(self, batchArray, solutionArray=None):
		return np.argsort(batchArray, axis=1)



