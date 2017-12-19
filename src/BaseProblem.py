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

	def accuracy(self, solution, batchArray):
		raise NotImplementedError

	def makeTargets(self, batchArray):
		# return decoder training inputs and one hot targets

		solution    = self.getSolution(batchArray)
		argArray    = self.getArgs(batchArray=batchArray, solutionArray=solution)
		high        = np.max(argArray) + 1
		oneHots     = np.eye(high)[np.squeeze(argArray, axis=2)]
		return solution, oneHots