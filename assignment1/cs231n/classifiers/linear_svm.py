import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        
        continue
      if margin > 0:
        loss += margin
        
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(np.square(W))

  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather than first computing the loss and then computing the derivative,    #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  correct_class_score = scores[range(num_train), y].reshape(num_train, -1)
  margin = scores - correct_class_score + 1
  margin[range(num_train), y] = 0
  margin_mask = margin > 0
  validmargin = margin[margin_mask]
  loss = np.sum(validmargin)/num_train + 0.5*reg*np.sum(np.square(W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

    #  The vectorization for this LinearSVM can be divided into two parts.
    #  All the calculations can be factorized into masking and multiplying.
    
    #  Masking respectively for y[i] == j and y[i]!= j
    #  And then multiplying the mask with X
  mask_matrix = np.ones([num_train, num_class])
  mask_matrix *= margin_mask
    # Masking using margin. Equivalent to the first half part of dW[:, j] += X[i].
  
  #  mask_matrix[range(num_train), y] = 0
    # Correcting all y[i] == j terms for summing minusing. 
  
  mask_matrix[range(num_train), y] = -np.sum(mask_matrix, axis=1)
    # Doing the summing minusing. Masking for all y[i] == j terms done.
  
  dW_x = X.T.dot(mask_matrix)
    # Vectorization Done.
  dW = reg*W + dW_x/num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
