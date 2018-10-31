import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  #print(num_train)
  #print(num_class)
  

  for i in xrange(num_train):
    temp = 0.0
    for j in xrange(num_class):
      temp += np.exp(np.dot(X[i],W[:,j]))
      if(j == y[i]):
        loss -= np.dot(X[i],W[:,j])
    loss += np.log(temp)
    
    for j in xrange(num_class):
      ind = 0
      if(j == y[i]):
      	ind = 1
      dW[:,j] += (np.exp(np.dot(X[i],W[:,j]))/temp - ind)*X[i]
  
  loss /= num_train
  loss += reg * np.sum(W*W) 
  dW /= num_train
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  # save loss
  scores = np.dot(X,W)
  stable_scores = scores - np.max(scores,axis = 1).reshape(-1,1)
  exp_stable_scores = np.exp(stable_scores)
  prob = exp_stable_scores/(np.sum(exp_stable_scores,axis = 1)).reshape(-1,1)
  ind = np.arange(num_train)
  correct_prob = prob[ind,y]
  loss = np.sum(-np.log(correct_prob))
  loss /= num_train
  loss += reg*np.sum(W**2)
	# save dW  
  dout = exp_stable_scores/(np.sum(exp_stable_scores,axis = 1)).reshape(-1,1)
  dout[ind,y] -= 1
  dW = np.dot(X.T,dout)
  dW /= num_train
  dW += 2*reg*W
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

