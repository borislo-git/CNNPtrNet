import tensorflow as tf
import configparser

class PointerNetwork(object):

	def __init__(self, configFile):

		self.loss = 0

		self.readConfig(configFile)
		self.makePlaceholders()
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

	def attention_mask(self, W_ref, W_q, v, enc_outputs, query, already_played_actions=None,
                      already_played_penalty=1e6):
		with tf.variable_scope("attention_mask"):
			u_i0s = tf.einsum('kl,itl->itk', W_ref, enc_outputs)
			u_i1s = tf.expand_dims(tf.einsum('kl,il->ik', W_q, query), 1)
			u_is = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s)) - already_played_penalty * already_played_actions
			return u_is, tf.nn.softmax(u_is)

	def CNNAttention(self, W, b, encOutputs, query, queryInputs, already_played_actions=None, already_played_penalty=1e6):
		'''
		Attention mechanism following "Convolutional Sequence to Sequence Learning" Gehring (2017)
		'''
		qshape = query.shape
		d                  = tf.einsum('kl,itl->itk', W, query) + queryInputs[:,:qshape[1],:] + b
		unscaledAttnLogits = tf.einsum('bjl,bic->bij', encOutputs, d) - already_played_penalty * already_played_actions[:, :qshape[1], :]
		attnLogits         = tf.nn.softmax(unscaledAttnLogits)
		return unscaledAttnLogits, attnLogits

	def makePlaceholders(self):

		with tf.variable_scope('Placeholders'):
			self.train     = tf.placeholder(tf.bool)

			# batch size x time steps x channels
			self.rawInputs = tf.placeholder(tf.float32, [None, None, self.inputSize])
			self.W_embed   = tf.get_variable('Input_Embed_Matrix', shape=[self.embeddingSize, self.inputSize],
	                                       initializer=tf.truncated_normal_initializer(stddev=self.stddev))

			self.embeddedInputs = tf.einsum('kl,itl->itk', self.W_embed, self.rawInputs)

			self.targets         = tf.placeholder(tf.float32, [None, None, None])
			self.targetInputs    = tf.placeholder(tf.float32, [None, None, self.inputSize])
			self.embeddedTargets = tf.einsum('kl,itl->itk', self.W_embed, self.targetInputs)

	def CNNLayer(self, inputs, filters, dilation, name, residual=False, kernel_size=2, layertype='gau', causal=False):
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
			numPad   = dilation*(kernel_size-1)
			paddings = tf.constant([[0, 0], [numPad, 0], [0, 0]])
			inputs   = tf.pad(inputs, paddings, mode='CONSTANT', constant_values=0)

		# apply dropout to input layer only
		if name == 0:
			inputs = tf.layers.dropout(inputs = inputs,
									rate      = self.dropoutRate,
									training  = self.train)

		if layertype == 'gau':
		    gate = tf.layers.conv1d(inputs        = inputs,
		                            filters       = filters,
		                            kernel_size   = kernel_size,
		                            dilation_rate = dilation,
		                            padding       = padding,
		                            activation    = tf.sigmoid,
		                            trainable     = True,
                                    name          = gateName)

		    filt = tf.layers.conv1d(inputs        = inputs,
		                            filters       = filters,
		                            kernel_size   = kernel_size,
		                            dilation_rate = dilation,
		                            padding       = padding,
		                            activation    = tf.tanh,
		                            trainable     = True,
                                    name          = filtName)
		    out = gate * filt

		elif layertype == 'relu':
			out = tf.layers.conv1d(inputs         = inputs,
		                            filters       = filters,
		                            kernel_size   = kernel_size,
		                            dilation_rate = dilation,
		                            padding       = padding,
		                            activation    = tf.relu,
		                            trainable     = True)
  
		if residual:
			if not causal:
				return inputs + out
			else:
				return inputs[:, numPad:, :] + out
		else:
			return out

	def makeEncoder(self):
		'''
		Set up encoder
		'''

		with tf.variable_scope('Encoder'):
			numFilters   = self.hiddenSize/self.encNumDilationLayer
			self.encConv = []
			self.encConv.append(self.CNNLayer(self.embeddedInputs, filters=numFilters, dilation=1, name = 0))
			# make the other layers
			factors = [2, 4, 8, 16, 32, 64]
			for layerNum in range(0, self.encNumDilationLayer-1):
				self.encConv.append(self.CNNLayer(self.encConv[-1], filters=numFilters, dilation=factors[layerNum], residual=True, name=layerNum+1))

			self.encOutputs = tf.concat(self.encConv, axis=2)

			# avg pooling to mimic lstm state
			# self.encFinalState_c = tf.reduce_mean(self.encOutputs, axis=1)

			# for "LSTM output (h)" do tanh(c)
			# self.encFinalState = tf.nn.rnn_cell.LSTMStateTuple(c = self.encFinalState_c, h = tf.nn.tanh(self.encFinalState_c))

	def makeDecoder(self):
		'''
		Set up decoder
		'''

		with tf.variable_scope('Decoder') as scope:
			numFilters   = self.hiddenSize
			self.decConv = []
			self.W_attn  = []
			self.b_attn  = []

			self.W_attn.append(tf.get_variable('W_attn_0', shape=[self.hiddenSize, self.hiddenSize], initializer=tf.truncated_normal_initializer))
			self.b_attn.append(tf.get_variable('b_attn_0', shape=[self.hiddenSize], initializer=tf.truncated_normal_initializer))

			self.decConv.append(self.CNNLayer(self.embeddedTargets, filters=numFilters, dilation=1, causal=True, name=0))

			# penalize the attention on things already selected
			already_played_actions = tf.concat([tf.reduce_sum(self.targets[:, :time, :], axis=1, keepdims=True) for time in range(1, self.maxTimeSteps)], axis=1)
			already_played_actions = tf.pad(already_played_actions, paddings=[[0, 0], [1, 0], [0, 0]])
			
			# Training
			unscaledAttnLogits, _ = self.CNNAttention(W=self.W_attn[-1], b=self.b_attn[-1], 
													encOutputs=self.encOutputs, 
													query=self.decConv[-1], 
													queryInputs=self.embeddedTargets, 
													already_played_actions = already_played_actions)

			self.loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=unscaledAttnLogits))
			
			# factors = [2, 4, 8, 16, 32, 64]
			# for layerNum in range(0, self.decNumDilationLayer-1):
			# 	self.decConv.append(self.CNNLayer(self.decConv[-1], filters=numFilters, dilation=factors[layerNum], residual=True, causal=True, name=layerNum+1))
			
			# Inference
			scope.reuse_variables()
			self.decoderOutputs = []

			# penalize the attention on things already selected
			already_played_actions = tf.zeros(shape=[self.batchSize, 1, self.maxTimeSteps], dtype=tf.float32)

			self.start          = tf.constant(dtype=tf.float32, value=-1, shape=(self.batchSize, 1, self.inputSize))
			self.inferenceInput = tf.einsum('kl,itl->itk', self.W_embed, self.start)
			for time in range(self.maxTimeSteps):

				query = self.CNNLayer(self.inferenceInput, filters=numFilters, dilation=1, causal=True, name=0)

				# Inference
				_, attnLogits = self.CNNAttention(W=self.W_attn[-1], b=self.b_attn[-1], 
					encOutputs=self.encOutputs, 
					query=query, 
					queryInputs=self.inferenceInput,
					already_played_actions=already_played_actions)

				# the next output is just from the newest time
				self.decoderOutputs.append(tf.argmax(attnLogits[:,-1,:], axis=1))
				
				totalActions = tf.expand_dims(tf.one_hot(self.decoderOutputs[-1], depth=self.maxTimeSteps) + already_played_actions[:, time, :],  dim=1) 
				newInput     = tf.expand_dims(tf.einsum('itk,it->ik', self.embeddedInputs, 
					tf.one_hot(self.decoderOutputs[-1], depth=self.maxTimeSteps)), dim=1)

				already_played_actions = tf.concat([already_played_actions, totalActions], axis=1)

				self.inferenceInput = tf.concat([self.inferenceInput, newInput], axis=1)

	def makeOptimizer(self):
		'''
		Set up the optimizer and also add regularization and control the learning rate
		'''

		# add L2 regularization on kernels
		for v in tf.trainable_variables():
			if 'kernel' in v.name:
				self.loss += self.l2Reg * tf.nn.l2_loss(v)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		self.trainOp   = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

if __name__ == '__main__':
	ntr = PointerNetwork('cfg')
