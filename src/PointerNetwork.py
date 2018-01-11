import tensorflow as tf
import configparser

class PointerNetwork(object):
	'''
	Pointer Network

	RNN parts are based on class project implementation

	The output of this network are the a list of indicies corresponding to an element in the input
	'''

	def __init__(self, configFile):
		'''
		Read in paramters from the config file
		'''

		self.loss       = 0
		self.globalStep = tf.Variable(0, trainable=False)
		self.readConfig(configFile)

	def makeGraph(self):
		'''
		makes the graph
		'''
		with tf.variable_scope('Pointer_Network'):
			self.makePlaceholders()

			self.makeEmbeddings()

			if self.encType == 'cnn':
				self.makeCNNEncoder()
			elif self.encType == 'rnn':
				self.makeRNNEncoder()
			else:
				raise ValueError('Encoder type must be cnn or rnn')

			if self.decType == 'cnn':
				self.makeCNNDecoder()
			elif self.decType == 'rnn':
				self.makeRNNDecoder()
			else:
				raise ValueError('Decoder type must be cnn or rnn')

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
		self.encType             = self.cparser.get('ENCODER', 'TYPE')

		self.decKernelSize       = self.cparser.getint('DECODER', 'KERNEL_SIZE')
		self.decNumDilationLayer = self.cparser.getint('DECODER', 'NUM_DILATION_LAYERS')		
		self.decType             = self.cparser.get('DECODER', 'TYPE')
		self.glimpse             = self.cparser.getboolean('DECODER', 'GLIMPSE')

		self.l2Reg       = self.cparser.getfloat('TRAIN', 'L2_REG')
		self.dropoutRate = self.cparser.getfloat('TRAIN', 'DROPOUT_RATE')
		self.clipNorm    = self.cparser.getfloat('TRAIN', 'CLIP_NORM_THRESHOLD')

	def RNNAttention(self, W_ref, W_q, v, attnInputs, query, alreadySelected=None, alreadySelectedPenalty=1e6):
		'''
		Attention mechanism in Vinyals (2015)

		attnInputs are the states over which to attend over
		'''
		with tf.variable_scope("RNN_Attention"):
			u_i0s              = tf.einsum('kl,itl->itk', W_ref, attnInputs)
			u_i1s              = tf.expand_dims(tf.einsum('kl,il->ik', W_q, query), 1)
			unscaledAttnLogits = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s))

			if alreadySelected is not None:
				unscaledAttnLogits -= alreadySelectedPenalty * alreadySelected

			attnLogits         = tf.nn.softmax(unscaledAttnLogits)
			context            = tf.einsum('bi,bic->bc', attnLogits, attnInputs)
			return unscaledAttnLogits, attnLogits, context

	def CNNAttention(self, W, b, attnInputs, query, queryInputs, alreadySelected=None, alreadySelectedPenalty=1e6):
		'''
		Attention mechanism following "Convolutional Sequence to Sequence Learning" Gehring (2017)
		'''
		with tf.variable_scope("CNN_Attention"):
			qshape = query.shape

			# queryInputs has previous context concatenated and also time dimension is dynamic during inference
			d                  = tf.einsum('kl,itl->itk', W, query) + queryInputs[:,:qshape[1],:qshape[2]] + b

			unscaledAttnLogits = tf.einsum('bjl,bic->bij', attnInputs, d) 

			if alreadySelected is not None:
				unscaledAttnLogits -= alreadySelectedPenalty * alreadySelected[:, :qshape[1], :]

			attnLogits         = tf.nn.softmax(unscaledAttnLogits)
			context            = tf.einsum('bij,bjc->bic', attnLogits, attnInputs)
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


	def CNNLayer(self, inputs, filters, dilation, name, residual=False, kernelSize=2, causal=False):
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
		if name == 0:
			inputs = tf.layers.dropout(inputs   = inputs,
									   rate     = self.dropoutRate,
									   training = self.train)

		# gated activation unit
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

		# residual does not add the input context vector which is concatenated after the original outputs of previous layer
		if residual:
			if not causal:
				return inputs[:, :, :out.shape[2]] + out
			else:
				return inputs[:, numPad:, :out.shape[2]] + out
		else:
			return out

	def makeCNNEncoder(self):
		'''
		Set up dilated CNN encoder
		'''

		with tf.variable_scope('Encoder'):
			# numFilters   = self.hiddenSize/self.encNumDilationLayer
			numFilters = self.hiddenSize

			self.encConv = [self.embeddedInputs]

			# make the other layers
			factors = [1, 2, 4, 1, 1 ,1, 1]
			for layerNum in range(0, self.encNumDilationLayer):
				useRes = (layerNum!=0)
				self.encConv.append(self.CNNLayer(self.encConv[-1], 
									kernelSize = self.encKernelSize, 
									filters    = numFilters, 
									dilation   = factors[layerNum], 
									residual   = useRes, 
									name       = layerNum))

			# self.encOutputs = tf.concat(self.encConv[1:], axis=2)
			self.encOutputs = self.encConv[-1]

			self.attnInputs = self.encConv[0]

			# make LSTM states for RNN decoder
			if self.decType == 'rnn':
				encFinalState_c    = tf.reduce_mean(self.encOutputs, axis=1)
				self.encFinalState = tf.nn.rnn_cell.LSTMStateTuple(c = encFinalState_c,
																   h = tf.nn.tanh(encFinalState_c))

	def makeCNNDecoder(self):
		'''
		Set up causal dilated CNN decoder
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
														           attnInputs      = self.attnInputs, 
														           query           = self.decConv[-1], 
														           queryInputs     = self.decConv[-2],
														           alreadySelected = alreadySelected)	


				# concatenate context vector 
				# self.decConv[-1] = tf.concat([self.decConv[-1], context], axis=2)
				self.decConv[-1] += context

			# glimpse
			if self.glimpse:
				unscaledAttnLogits, _, _ = self.CNNAttention(W             = self.W_attn[-1, :, :], 
														     b               = self.b_attn[-1, :], 
														     attnInputs      = self.attnInputs, 
														     query           = context, 
														     queryInputs     = self.embeddedTargets, 
														     alreadySelected = alreadySelected)		

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
				queryInp = [self.inferenceInput]
				for layerNum in range(0, self.decNumDilationLayer):
					query = self.CNNLayer(queryInp[-1], 
										  filters    = numFilters, 
										  kernelSize = self.decKernelSize,
										  dilation   = factors[layerNum], 
										  residual   = True, 
										  causal     = True, 
										  name       = layerNum)

					_, attnLogits, context = self.CNNAttention(W               = self.W_attn[layerNum, :, :], 
															   b               = self.b_attn[layerNum, :], 
														       attnInputs      = self.attnInputs, 
														       query           = query, 
														       queryInputs     = queryInp[-1],
														       alreadySelected = alreadySelected)

					# queryInp = tf.concat([query, context], axis=2)
					queryInp.append(query + context)

				# glimpse
				if self.glimpse:
					_, attnLogits, _ = self.CNNAttention(W               = self.W_attn[-1, :, :], 
													     b               = self.b_attn[-1, :], 
												         attnInputs      = self.attnInputs, 
												         query           = context, 
												         queryInputs     = queryInp[-2], 
												         alreadySelected = alreadySelected)	

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

	def makeRNNEncoder(self):
		'''
		Set up LSTM Encoder
		'''

		with tf.variable_scope('Encoder'):
			encRNNCell = tf.nn.rnn_cell.LSTMCell(self.hiddenSize)
			self.encOutputs, self.encFinalState = tf.nn.dynamic_rnn(cell   = encRNNCell,
																	inputs = self.embeddedInputs,
																	dtype  = tf.float32)
			self.attnInputs = self.embeddedInputs

	def makeRNNDecoder(self):
		'''
		Set up LSTM Decoder
		'''

		with tf.variable_scope('Decoder') as scope:
			decRNNCell = tf.nn.rnn_cell.LSTMCell(self.hiddenSize)
			# decRNNCell = tf.contrib.rnn.ResidualWrapper(decRNNCell)

			# Define attention weights
			with tf.variable_scope("Attention_Weights"):
				W_ref = tf.Variable(tf.random_normal([self.hiddenSize, self.hiddenSize], stddev=self.stddev),
						name='W_ref')
				W_q = tf.Variable(tf.random_normal([self.hiddenSize, self.hiddenSize], stddev=self.stddev),
						name='W_q')
				v = tf.Variable(tf.random_normal([self.hiddenSize], stddev=self.stddev),
						name='v')

			# Training
			decoderInput = tf.tile(tf.Variable(tf.random_normal([1, self.embeddingSize])), [self.batchSize, 1])
			decoderState = self.encFinalState

			alreadySelected = tf.zeros(shape=[self.batchSize, self.maxTimeSteps], dtype=tf.float32)
			decoderInputs = [decoderInput]
			for t in range(self.maxTimeSteps):
				decOutput, decoderState = decRNNCell(inputs=decoderInput,
				      								 state=decoderState)

				unscaledAttnLogits, _, context = self.RNNAttention(W_ref, W_q, v, self.attnInputs, decOutput, alreadySelected=alreadySelected)
				# glimpse
				if self.glimpse:
					unscaledAttnLogits, _, _ = self.RNNAttention(W_ref, W_q, v, self.attnInputs, context, alreadySelected=alreadySelected)

				self.loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets[:, t, :], logits=unscaledAttnLogits))
				
				# feed in exact solution as input
				decoderInput = tf.einsum('itk,it->ik', self.embeddedInputs, self.targets[:, t, :])
				decoderInputs.append(decoderInput)
				alreadySelected += self.targets[:, t, :]

			# Inference
			scope.reuse_variables()
			decoderInput   = tf.tile(tf.Variable(tf.random_normal([1, self.embeddingSize])), [self.batchSize, 1])
			decoderState   = self.encFinalState
			self.decoderOutputs = []
			alreadySelected = tf.zeros(shape=[self.batchSize, self.maxTimeSteps], dtype=tf.float32)
			for _ in range(self.maxTimeSteps):
				decOutput, decoderState = decRNNCell(inputs=decoderInput,
				      								 state=decoderState)

				_, attnLogits, context = self.RNNAttention(W_ref, W_q, v, self.attnInputs, decOutput, alreadySelected=alreadySelected)
				# glimpse
				if self.glimpse:
					_, attnLogits, _ = self.RNNAttention(W_ref, W_q, v, self.attnInputs, context, alreadySelected=alreadySelected)

				self.decoderOutputs.append(tf.argmax(attnLogits, axis=1))

				# feed in output as next input
				decoderInput     = tf.einsum('itk,it->ik', self.embeddedInputs, tf.one_hot(self.decoderOutputs[-1], depth=self.maxTimeSteps))
				alreadySelected += tf.one_hot(self.decoderOutputs[-1], depth=self.maxTimeSteps)

	def makeOptimizer(self):
		'''
		Set up the optimizer and also add regularization and control the learning rate
		'''

		# add L2 regularization on kernels
		for v in tf.trainable_variables():
			if 'kernel' in v.name:
				self.loss += self.l2Reg * tf.nn.l2_loss(v)

		starter_learning_rate = 1e-2
		learning_rate         = tf.train.exponential_decay(starter_learning_rate, self.globalStep,
                                           500, 0.96, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

		# gradient norm clipping
		# gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
		# gradients, _ = tf.clip_by_global_norm(gradients, self.clipNorm)
		# self.trainOp  = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.globalStep)

		self.trainOp = self.optimizer.minimize(self.loss, global_step=self.globalStep)
	def printVarsStats(self):
		'''
		Print the names and total number of variables in graph
		'''

		numVars = 0
		for v in tf.trainable_variables():
			print(v.name)
			tmp = 1
			for dim in v.shape:
				tmp *= dim.value
			numVars += tmp
		print('Number of variables: '+str(numVars))


if __name__ == '__main__':
	ntr = PointerNetwork('../cfg')
	ntr.makeGraph()
	ntr.printVarsStats()
