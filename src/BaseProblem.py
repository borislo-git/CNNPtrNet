# base problem class for testing pointer network
import numpy as np
class BaseProblem(object):

	def __init__(self):
		pass

	def makeBatch(self, params):
		raise NotImplementedError

	def getArgs(self, batchArray, solutionArray=None):
		raise NotImplementedError

	def getSolution(self, batchArray):
		raise NotImplementedError

	def makeTargets(self, batchArray):
		# return decoder training inputs and one hot targets

		# prepend -1 as the start token 
		solution    = self.getSolution(batchArray)
		argArray    = self.getArgs(batchArray)
		high        = np.max(argArray) + 1
		oneHots     = np.eye(high)[np.squeeze(argArray, axis=2)]
		return np.insert(solution, obj=0, values=-1, axis=1), oneHots