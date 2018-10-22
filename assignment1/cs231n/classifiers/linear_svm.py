import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  #10
  num_classes = W.shape[1]
  #500
  num_train = X.shape[0]
  loss = 0.0
  
  lossMatrix = np.zeros((num_train,num_classes))   
  for i in xrange(num_train):
    # X : 500 X 3073 W: 3073 x 10 -> 1x3073 dot 3073x10 = 1x10       
    scores = X[i].dot(W)
    
    correct_class_score = scores[y[i]]
    #print(scores[y[i]])
    #print(i)
    #print(scores[y[i]])
    #loss_temp = 0.0
    for j in xrange(num_classes):
      #print(j)        
      if j == y[i]:
        continue
      #print(scores[j])
      margin = scores[j] - correct_class_score + 1 # note delta = 1  
             
      if margin > 0:
        loss += margin
        #loss_temp += 1
        #lossMatrix[i,j]= 1
        dW[:,j] += X[i,:].T
        dW[:,y[i]] += (-1) * X[i,:].T 
        
    #lossMatrix[i,y[i]] = (-1)*(np.sum(lossMatrix[i,:]))
    #dW[:,y[i]] += (-1)*(np.sum(lossMatrix[i,:]))* X[i,:]   
    #dW[:,y[i]] += (-1) * loss_temp * X[i,:]   
   

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #dW = (X.T).dot(lossMatrix)/num_train  + reg * W 
  dW = (dW/num_train) + reg * W
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
  y_correct = np.zeros(y.shape[0])
  margin_matrix = np.zeros((X.shape[0] , W.shape[1])) 
    
  #np.choose(select_id, input_array.T)  
  #y_correct = X.dot(W)[np.arange(len(X.dot(W))),y]
  y_correct = np.choose(y,X.dot(W).T)
  #(500,)
  #print(y_correct.shape)
  #y_correct.reshape((y_correct.shape[0] , 1))
  
  margin_matrix = X.dot(W) - np.vstack(y_correct)  + 1
  
  #loss = np.sum(margin_matrix[(margin_matrix > 0) & (margin_matrix != 1)])/ X.shape[0] 
  loss = (np.sum(margin_matrix[(margin_matrix > 0)]) -np.sum(np.choose(y,margin_matrix.T))) /X.shape[0]   
    
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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
