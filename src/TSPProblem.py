import numpy as np
from itertools import permutations
import math
from .BaseProblem import BaseProblem


class TSPProblem(BaseProblem):


	def dist(self, x1, x2):
		'''
		Euclidean distance
		'''
		return math.sqrt(sum([(x1[i] - x2[i])**2 for i in range(len(x1))]))

	def pathLength(self, path):
		'''
		Return the total path length given path (a list of points)
		'''
		return sum([self.dist(path[i], path[i+1]) for i in range(len(path)-1)])

	def makeBatch(self, params):
		'''
		Cities live in the unit square
		'''
		return np.random.rand(params['batchSize'], params['numCities'], 2)

	def getSolution(self, batchArray):
		'''
		Given np array [batchSize x numCities x dimension] find the esolution for each set 

		taken from https://codereview.stackexchange.com/questions/81865/travelling-salesman-using-brute-force-and-heuristics
		'''
		output = []

		# brute force for small problems
		if batchArray.shape[1] < 8:
			for setNum in range(batchArray.shape[0]):
				thisSet = batchArray[setNum, :, :].tolist()
				output.append(min([perm for perm in permutations(thisSet) if perm[0] == thisSet[0]], key=self.pathLength))

		# nearest neighbors for everything else
		else:
			for setNum in range(batchArray.shape[0]):
				thisSet = batchArray[setNum, :, :].tolist()
				must_visit = thisSet[1:]
				path = [thisSet[0]]
				while must_visit:
					nearest = min(must_visit, key=lambda x: self.dist(path[-1], x))
					path.append(nearest)
					must_visit.remove(nearest)
				output.append(path)
		return np.array(output)

	def getArgs(self, batchArray, solutionArray):
		output = []
		for pathNum in range(batchArray.shape[0]):
			indexForThisPath = []
			for pointNum in range(solutionArray.shape[1]):
				indexForThisPath.append(np.where(np.all(batchArray[pathNum,:,:]==solutionArray[pathNum,pointNum,:],axis=1))[0].tolist())
			output.append(indexForThisPath)

		return np.array(output)

	def accuracy(self, solution, batchArray):
		'''
		Compute the accuracy of the solutions defined as the average(network solution path length/getSolution path length)
		'''
		pathLengths = []
		for pathNum in range(batchArray.shape[0]):
			solutionPath = batchArray[pathNum,:,:][solution[pathNum]].tolist()

			inputSet     = self.getSolution(batchArray)[pathNum,:,:].tolist()

			pathLengths.append(self.pathLength(solutionPath)/self.pathLength(inputSet))

		return np.mean(pathLengths)
