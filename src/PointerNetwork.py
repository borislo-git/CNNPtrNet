import tensorflow as tf
import configparser

class PointerNetwork(object):

	def __init__(self, configFile):

		self.loss = 0

		self.readConfig(configFile)
		self.makePlaceholders()
		self.makeEmbeddings()
		self.makeEncoder()
		self.makeDecoder()
		self.makeOptimizer()

	def readConfig(self, configFile):
		'''
		Use config parser to get model parameters
		'''

		self.cparser = configparser.ConfigParser()
		self.cparser.read(configFile)
		self.batchSize     = self.cparser.getint('INPUTS', 'BATCH_SIZE')
		self.embeddingSize = self.cparser.getint('INPUTS', 'EMBED_SIZE')
		self.inputSize     = self.cparser.getint('INPUTS', 'INPUT_SIZE')
		self.maxTimeSteps  = self.cparser.getint('INPUTS', 'MAX_TIME')
		self.stddev        = self.cparser.getfloat('INPUTS', 'STD_DEV')

		self.encKernelSize       = self.cparser.getint('ENCODER', 'KERNEL_SIZE')
		self.encNumDilationLayer = self.cparser.getint('ENCODER', 'NUM_DILATION_LAYERS')
		self.hiddenSize          = self.cparser.getint('ENCODER', 'HIDDEN_SIZE')

		self.decKernelSize       = self.cparser.getint('DECODER', 'KERNEL_SIZE')
		self.decNumDilationLayer = self.cparser.getint('DECODER', 'NUM_DILATION_LAYERS')		

		self.l2Reg       = self.cparser.getfloat('TRAIN', 'L2_REG')
		self.dropoutRate = self.cparser.getfloat('TRAIN', 'DROPOUT_RATE')


	def CNNAttention(self, W, b, encOutputs, query, queryInputs, alreadySelected=None, alreadySelectedPenalty=1e6):
		'''
		Attention mechanism following "Convolutional Sequence to Sequence Learning" Gehring (2017)
		'''
		qshape = query.shape
		d                  = tf.einsum('kl,itl->itk', W, query) + queryInputs[:,:qshape[1],:] + b
		unscaledAttnLogits = tf.einsum('bjl,bic->bij', encOutputs, d) - alreadySelectedPenalty * alreadySelected[:, :qshape[1], :]
		attnLogits         = tf.nn.softmax(unscaledAttnLogits)
		context            = tf.einsum('bij,bjc->bic', attnLogits, encOutputs)
		return unscaledAttnLogits, attnLogits, context

	def makePlaceholders(self):

		with tf.variable_scope('Placeholders'):
			self.train     = tf.placeholder(tf.bool)

			# batch size x time steps x channels
			self.rawInputs = tf.placeholder(tf.float32, [None, None, self.inputSize])


			self.targets         = tf.placeholder(tf.float32, [None, None, None])
			self.targetInputs    = tf.placeholder(tf.float32, [None, None, self.inputSize])

	def makeEmbeddings(self):
		'''
		Small embedding layer
		'''
		with tf.variable_scope('Embedding') as scope:
			self.W_embed   = tf.get_variable('Input_Embed_Matrix', shape=[self.embeddingSize, self.inputSize],
	                                       initializer=tf.truncated_normal_initializer(stddev=self.stddev))

			self.embeddedInputs  = tf.nn.leaky_relu(tf.einsum('kl,itl->itk', self.W_embed, self.rawInputs))

			self.embeddedTargets = tf.nn.leaky_relu(tf.einsum('kl,itl->itk', self.W_embed, self.targetInputs))


	def CNNLayer(self, inputs, filters, dilation, name, residual=False, kernelSize=2, layertype='gau', causal=False):
		'''
		Generates a CNN layer
		'''
		gateName = 'gate_'+str(name)
		filtName = 'filter_'+str(name)
		if not causal:
			padding = 'SAME'
		else:
			padding  = 'VALID'
			inpShape = inputs.shape

			# zero padding for casual convolutions
			numPad   = dilation*(kernelSize-1)
			paddings = tf.constant([[0, 0], [numPad, 0], [0, 0]])
			inputs   = tf.pad(inputs, paddings, mode='CONSTANT', constant_values=0)

		# apply dropout to input layer only
		# if name == 0:
		# 	inputs = tf.layers.dropout(inputs   = inputs,
		# 							   rate     = self.dropoutRate,
		# 							   training = self.train)

		if layertype == 'gau':
		    gate = tf.layers.conv1d(inputs        = inputs,
		                            filters       = filters,
		                            kernel_size   = kernelSize,
		                            dilation_rate = dilation,
		                            padding       = padding,
		                            activation    = tf.sigmoid,
		                            trainable     = True,
                                    name          = gateName)

		    filt = tf.layers.conv1d(inputs        = inputs,
		                            filters       = filters,
		                            kernel_size   = kernelSize,
		                            dilation_rate = dilation,
		                            padding       = padding,
		                            activation    = tf.tanh,
		                            trainable     = True,
                                    name          = filtName)
		    out = gate * filt

		elif layertype == 'relu':
			out = tf.layers.conv1d(inputs         = inputs,
		                            filters       = filters,
		                            kernel_size   = kernelSize,
		                            dilation_rate = dilation,
		                            padding       = padding,
		                            activation    = tf.relu,
		                            trainable     = True)
  
		if residual:
			if not causal:
				return inputs[:, :, :out.shape[2]] + out
			else:
				return inputs[:, numPad:, :out.shape[2]] + out
		else:
			return out

	def makeEncoder(self):
		'''
		Set up encoder
		'''

		with tf.variable_scope('Encoder'):
			numFilters   = self.hiddenSize/self.encNumDilationLayer
			self.encConv = [self.embeddedInputs]

			# make the other layers
			factors = [1, 2, 4, 8, 16, 32, 64]
			for layerNum in range(0, self.encNumDilationLayer):
				useRes = (layerNum!=0)
				self.encConv.append(self.CNNLayer(self.encConv[-1], 
									kernelSize = self.encKernelSize, 
									filters    = numFilters, 
									dilation   = factors[layerNum], 
									residual   = useRes, 
									name       = layerNum))

			self.encOutputs = tf.concat(self.encConv, axis=2)

	def makeDecoder(self):
		'''
		Set up decoder
		'''

		with tf.variable_scope('Decoder') as scope:

			# convolution layer parameters
			numFilters   = self.hiddenSize
			factors = [1, 2, 4, 8, 16, 32, 64]

			# attention weights
			self.W_attn  = tf.get_variable('W_attn_0', shape=[self.decNumDilationLayer, self.hiddenSize, self.hiddenSize], initializer=tf.truncated_normal_initializer)
			self.b_attn  = tf.get_variable('b_attn_0', shape=[self.decNumDilationLayer, self.hiddenSize], initializer=tf.truncated_normal_initializer)

			# penalize the attention on things already selected
			alreadySelected = tf.concat([tf.reduce_sum(self.targets[:, :time, :], axis=1, keepdims=True) for time in range(1, self.maxTimeSteps)], axis=1)
			alreadySelected = tf.pad(alreadySelected, paddings=[[0, 0], [1, 0], [0, 0]])
			
			# stacked CNN
			self.decConv = [self.embeddedTargets]
			for layerNum in range(0, self.decNumDilationLayer):
				self.decConv.append(self.CNNLayer(self.decConv[-1], 
												  filters    = numFilters, 
												  kernelSize = self.decKernelSize,
												  dilation   = factors[layerNum], 
												  residual   = True, 
												  causal     = True, 
												  name       = layerNum))

				unscaledAttnLogits, _, context = self.CNNAttention(W               = self.W_attn[layerNum, :, :], 
																   b               = self.b_attn[layerNum, :], 
														           encOutputs      = self.encOutputs, 
														           query           = self.decConv[-1], 
														           queryInputs     = self.embeddedTargets, 
														           alreadySelected = alreadySelected)	

				# concatenate context vector 
				self.decConv[-1] = tf.concat([self.decConv[-1], context], axis=2)

			self.loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=unscaledAttnLogits))

			# Inference
			scope.reuse_variables()
			self.decoderOutputs = []

			# penalize the attention on things already selected
			alreadySelected = tf.zeros(shape=[self.batchSize, 1, self.maxTimeSteps], dtype=tf.float32)

			self.start          = tf.constant(dtype=tf.float32, value=-1, shape=(self.batchSize, 1, self.inputSize))
			self.inferenceInput = tf.einsum('kl,itl->itk', self.W_embed, self.start)
			for time in range(self.maxTimeSteps):

				# stacked CNN
				query = self.inferenceInput
				for layerNum in range(0, self.decNumDilationLayer):
					query = self.CNNLayer(query, 
										  filters    = numFilters, 
										  kernelSize = self.decKernelSize,
										  dilation   = factors[layerNum], 
										  residual   = True, 
										  causal     = True, 
										  name       = layerNum)

					_, attnLogits, context = self.CNNAttention(W               = self.W_attn[layerNum, :, :], 
															   b               = self.b_attn[layerNum, :], 
														       encOutputs      = self.encOutputs, 
														       query           = query, 
														       queryInputs     = self.embeddedTargets, 
														       alreadySelected = alreadySelected)	

					query = tf.concat([query, context], axis=2)

				# the next output is just from the newest time from last layer
				self.decoderOutputs.append(tf.argmax(attnLogits[:,-1,:], axis=1))
				
				# make a vector of length maxTimeSteps where totalActions[i] = 1 if i has been pointed to by this step
				totalActions = tf.expand_dims(tf.one_hot(self.decoderOutputs[-1], depth=self.maxTimeSteps) + alreadySelected[:, time, :],  dim=1) 
				newInput     = tf.expand_dims(tf.einsum('itk,it->ik', self.embeddedInputs, 
					tf.one_hot(self.decoderOutputs[-1], depth=self.maxTimeSteps)), dim=1)

				# update the set of already selected indicies for attention
				alreadySelected = tf.concat([alreadySelected, totalActions], axis=1)

				# next input
				self.inferenceInput = tf.concat([self.inferenceInput, newInput], axis=1)

	def makeOptimizer(self):
		'''
		Set up the optimizer and also add regularization and control the learning rate
		'''

		# add L2 regularization on kernels
		for v in tf.trainable_variables():
			if 'kernel' in v.name:
				self.loss += self.l2Reg * tf.nn.l2_loss(v)

		global_step           = tf.Variable(0, trainable=False)
		starter_learning_rate = 1e-2
		learning_rate         = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           500, 0.96, staircase=True)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.trainOp   = self.optimizer.minimize(self.loss, global_step=global_step)

if __name__ == '__main__':
	ntr = PointerNetwork('../cfg')

	for v in tf.trainable_variables():
		print(v.name)
