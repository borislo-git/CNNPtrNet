import numpy as np
import knapsack
from .BaseProblem import BaseProblem

class KnapsackProblem(BaseProblem):

	def makeBatch(self, params):
		size = np.random.randint(low  = params['lowSize'],
								 high = params['highSize']+1,
								 size = [params['batchSize'], params['length'], 1])
		weight = np.random.randint(low  = params['lowWeight'],
								 high = params['highWeight']+1,
								 size = [params['batchSize'], params['length'], 1])

		self.capacity = params['capacity']

		return np.concat([size, weight], axis=2)

	def getSolution(self, batchArray):
		out = []
		for batch in range(batchArray.shape[0]):
			out.append(knapsack.knapsack(batchArray[batch, :, 0].tolist(), batchArray[batch, :, 1].tolist()).solve(self.capacity))

		return np.array(out)

	def getArgs(self, batchArray, solutionArray=None):
		size = batchArray.shape
		totalArgs = []
		for batch in range(size[0]):
			batchArg = []
			numItemsInBatch = solutionArray[batch, :, :][0]
			for item in range(numItemsInBatch):
				if solutionArray[batch, item, 0] >= 0:
					batchArg.append(np.where(np.all(batchArray[batch,:,:]==solutionArray[batch,item,:], axis=1), axis=1)[0])
				else:
					batchArg.append(-1)

			totalArgs.append(batchArg)





