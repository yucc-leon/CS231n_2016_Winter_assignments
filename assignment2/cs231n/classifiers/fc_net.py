import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *

def wlayer(layer):
    """
        Returns the name of the parameter W that corresponds to the given layer.
    """
    return 'W{0}'.format(layer)

def blayer(layer):
    """
        Returns the name of the parameter b that corresponds to the given layer.
    """
    return 'b{0}'.format(layer)

def gammalayer(layer):
    """
        Returns the name of the parameter gamma that corresponds to the given layer.
    """
    return 'gamma{0}'.format(layer)
def betalayer(layer):
    """
        Returns the name of the parameter beta that corresponds to the given layer.
    """
    return 'beta{0}'.format(layer)

def affine_batch_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform followed by a Batch normalization.
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: parameters of the Batch Norm layer
    - bn_param: Dictionary with the following keys:
     - mode: 'train' or 'test'; required
     - eps: Constant for numeric stability
     - momentum: Constant for runnign mean / variance
     - running_mean: Array of shape (D,) giving running mean of features
     - running_var: Array of shape(D,) giving running variance of features
     
    Returns a tuple of:
    - out: Output from the Batch-Norm layer
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_batch_relu_backward(dout, cache):
    """
    Backward pass for the affine_relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale*np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    reg = self.reg
    
    hidOut, hidCache = affine_relu_forward(X, W1, b1)
    scores, cache = affine_forward(hidOut, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dloss_dscores = softmax_loss(scores, y)
    dloss_dhidOut, dW2, db2 = affine_backward(dloss_dscores, cache)
    _, dW1, db1 = affine_relu_backward(dloss_dhidOut, hidCache)
    
    loss += 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
    grads['W1'], grads['b1'] = dW1+reg*W1, db1
    grads['W2'], grads['b2'] = dW2+reg*W2, db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    prev_dim = input_dim
    for layer, hidden_dim in enumerate(hidden_dims):
        w = np.random.normal(scale=weight_scale, size=(prev_dim, hidden_dim))
        b = np.zeros((hidden_dim, ))
        self.params[wlayer(layer+1)] = w
        self.params[blayer(layer+1)] = b
        prev_dim = hidden_dim
        
        if self.use_batchnorm:
            self.params[gammalayer(layer+1)] = np.ones(hidden_dim)
            self.params[betalayer(layer+1)] = np.zeros(hidden_dim)
    w = np.random.normal(scale=weight_scale, size=(prev_dim, num_classes))
    b = np.zeros((num_classes, ))
    self.params[wlayer(self.num_layers)] = w
    self.params[blayer(self.num_layers)] = b
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    def forward(function, layer):
        """
            Auxiliary function that performs a forward pass of a given layer.
        """
        w = self.params[wlayer(layer)]
        b = self.params[blayer(layer)]
        return function(prev_input, w, b)
    
    
    def batchnorm_forward_aux(layer):
        """
            Auxiliary function that gets the required variables and performs a forward pass on given affine-batch-relu layers.
        """
        w = self.params[wlayer(layer)]
        b = self.params[blayer(layer)]
        gamma = self.params[gammalayer(layer)]
        beta = self.params[betalayer(layer)]
        bn_param = self.bn_params[layer-1]
        return affine_batch_relu_forward(prev_input, w, b, gamma, beta, bn_param)
    
    prev_input = X
    caches = [None] * self.num_layers
    for layer in range(1, self.num_layers):
        if self.use_batchnorm:
            prev_input, caches[layer-1] = batchnorm_forward_aux(layer)
        else:
            prev_input, caches[layer-1] = forward(affine_relu_forward, layer)
            
        
        if self.use_dropout:
            prev_input, c = dropout_forward(prev_input, self.dropout_param)
            caches[layer-1] += c
            
    scores, caches[self.num_layers - 1] = forward(affine_forward, self.num_layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    
    for layer in range(1, self.num_layers + 1):
        w = self.params[wlayer(layer)]
        loss += 0.5 * self.reg * np.sum(w * w)
    
    def backward(function, layer, dout, cache):
        """
            Auxiliary backward function that computes the gradient of the given layer, updates the gradient dictionary and returns the gradient.
        """
        df, dw, db = function(dout, cache)
        grads[wlayer(layer)] = dw + self.reg * self.params[wlayer(layer)]
        grads[blayer(layer)] = db
        return df
    def batchnorm_backward_aux(layer, dout, cache):
        """
            Auxiliary backward function that computes the gradient of the given affine-batch-relu layer, updates the gradient dictionary and returns the gradient.
        """
        df, dw, db, dgamma, dbeta = affine_batch_relu_backward(dout, cache)
        grads[wlayer(layer)] = dw + self.reg * self.params[wlayer(layer)]
        grads[blayer(layer)] = db
        grads[gammalayer(layer)] = dgamma
        grads[betalayer(layer)] = dbeta
        return df
    
    dout = backward(affine_backward, self.num_layers, dout, caches[self.num_layers - 1])
    for layer in reversed(range(1, self.num_layers)):
        c = caches[layer - 1]
        if self.use_dropout:
            dropout_cache = c[-2:]
            dout = dropout_backward(dout, dropout_cache)
            c = c[0:-2]
            
        if self.use_batchnorm:
            dout = batchnorm_backward_aux(layer, dout, c)
        else:
            dout = backward(affine_relu_backward, layer, dout, c)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
