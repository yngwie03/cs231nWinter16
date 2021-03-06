import numpy as np
from random import shuffle

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
  num_dim =  W.shape[0] 
 
  
  for i in xrange(num_train):
    f = X[i].dot(W)  
    f -= np.max(f)
    scores =np.exp(f)
    normalize = np.sum(scores)
    prob_scores = scores/normalize  
    #print(np.sum(prob_scores))
    loss_scores = (-1) * (np.log(prob_scores) )  
  
  
    for j in xrange(num_class):
      
      if j == y[i]:
        loss += loss_scores[j]
        prob_scores[j] -= 1
        
        dW += (X[i].reshape(num_dim,1)).dot(prob_scores.reshape(1,num_class))
      
 
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) 
  dW /= num_train
  dW += reg * W
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
  num_class = W.shape[1]
  num_dim =  W.shape[0]
  
  f = X.dot(W)
  
  c = (-1) * np.max(f,axis = 1)
   
  f += c[:,np.newaxis]

 
  
  #500 x 10
 
 
  weight_X_dot =  np.exp(f) 
  
  normalize = np.sum(weight_X_dot, axis=1)
  
   
  #prob_scores = weight_X_dot/normalize[:,None]
  #prob_scores = (weight_X_dot.T/weight_X_dot.sum(axis=1)).T
  prob_scores  =  weight_X_dot  /  np.sum(weight_X_dot, axis=1, keepdims=True)   
  
  #loss_scores = (-1) * (np.log(weight_X_dot) - np.log(np.sum(weight_X_dot, axis=1, keepdims=True) )) 
    
  loss_scores = (-1) * (f - np.log(np.sum(weight_X_dot, axis=1, keepdims=True) ))   
  
  loss = np.sum(loss_scores[np.arange(loss_scores.shape[0]), y] )
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) 
  prob_scores[np.arange(prob_scores.shape[0]), y] -= 1
  dW = X.T.dot(prob_scores) 
  dW /= num_train
  dW += reg * W  
  
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

