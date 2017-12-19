# CNNPtrNet

Extending Pointer Networks (Vinyals 2015) by replacing the RNNs with causal dilated CNNs, taking the attention mechanism from Convolution Sequence to Sequence (Gehring 2017)

The purpose is to make the network more suitable for set inputs, currently only test problem is sorting floats in [0,1).

Stack the encoder with dilated CNNs with residual. For the encoder, concatenate the outputs of each convolution layer; this would include the translational invariant information at the topmost layer and also some positional information so the decoder can knows where to point.

## Some Results
* CNN-Encoder with LSTM-Decoder performs the best

