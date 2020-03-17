from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        f_i = X[i].dot(W)
        f_i -= np.max(f_i) # for numrical stability
        f_i = np.exp(f_i)
        q_dist = np.sum(f_i) # "Distribution" of all classes
        p_dist = f_i[y[i]] # "Disribution" of the correct class
        loss_i = (-1) * np.log(p_dist / q_dist)
        loss += loss_i
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (-1) * (q_dist - p_dist) / q_dist * X[i]
            dW[:, j] += X[i] * f_i[j] / q_dist
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Check if the reg term is correct for softmax function.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    f = X.dot(W)
    f -= f.max(axis=1).reshape(num_train, 1) # for numrical stability
    f = np.exp(f)
    q_dist = np.sum(f, axis=1).reshape(num_train, 1) # "Distribution" of all classes
    p_dist = f[np.arange(num_train), y].reshape(num_train, 1) # "Disribution" of the correct class
    loss = (-1) * np.log(p_dist / q_dist)
    loss = np.sum(loss)
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    mask = f / q_dist.reshape(num_train, 1)
    mask[range(num_train), y] = ((-1) * (q_dist - p_dist) / q_dist).reshape(-1)
    dW = X.T.dot(mask)
    dW /= num_train
    dW += 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
