import numpy as np
import cvxopt
import random
from SMO import *
import sys

sys.dont_write_bytecode = True

class SVMTrainer:
	def __init__(self, c, eps, tol):
         self._c = c
         self._eps = eps
         self._tol = tol

	def train(self, X, y):
		"""Given the training features X with labels y, returns a SVM
			predictor representing the trained SVM.
	    """
		alpha = self.compute_alpha(X, y)
	   	return self.predictor(X, y, alpha)

	def kernel_matrix(self, X):
		"""
		Return the gram matrix of X
		"""
		n_samples, n_features = X.shape
		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
			for j in range(n_samples):
				K[i, j] = kernel(X[i,:], X[j,:])
		return K

	def predictor(self, X, y, alpha):
		"""
		Return a predictor after the training
		"""
		# Find all the support vectors
		support_vector_indices = alpha > 0
		# only save the support vectors to reduce the space
		weights = alpha[support_vector_indices]
		support_vectors = X[support_vector_indices]
		support_vector_labels = y[support_vector_indices]


		return SVMPredictor(weights,
			support_vectors,support_vector_labels)

	def compute_alpha(self, X, y):
            """
            n_samples, n_features = X.shape
            K = self.kernel_matrix(X)
		# Solves
		# min 1/2 x^T P x + q^T x
		# s.t.
		#  Gx \coneleq h
		#  Ax = b
            P = cvxopt.matrix(np.outer(y, y) * K)
            q = cvxopt.matrix(-1 * np.ones(n_samples))

		# -a_i \leq 0
		# TODO(tulloch) - modify G, h so that we have a soft-margin classifier
            G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h_std = cvxopt.matrix(np.zeros(n_samples))

		# a_i \leq c
            G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
            h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h = cvxopt.matrix(np.vstack((h_std, h_slack)))

            A = cvxopt.matrix(y, (1, n_samples))
            b = cvxopt.matrix(0.0)

            solution = cvxopt.solvers.qp(P, q, G, h, A, b)

		# Lagrange multipliers
            return np.ravel(solution['x'])
            """  
            
            SMOobj = SMO(X, y, self._c, self._tol, self._eps)
            SMOobj.SMO_main()
            return SMOobj.get_alpha()		
            
            
	
class SVMPredictor:
	""" Attributes:
	_weights: the weights of each support vector
	_support_vectors: the support vectors of this predictor
	_support_vector_labels: the labels of the support vectors
	"""

	def __init__(self, weights, support_vectors,
			support_vector_labels):
           self._weights = weights
           self._support_vectors = support_vectors
           self._support_vector_labels = support_vector_labels

	def predict(self, X):
         """
         Computes the SVM prediction on the given features x.
         """
         n_samples, n_features = X.shape
         labels = np.zeros(n_samples)
         for i in range(n_samples):
             result = 0
             for z_i, x_i, y_i in zip(self._weights,
                self._support_vectors,
                self._support_vector_labels):
                    result += z_i * y_i * kernel(x_i, X[i,:])
                    labels[i] = np.sign(result)
         return labels
			

def kernel(X1,X2):
    """ 
    Given two feature vectors, calculate the kernel of this two features
    Here, we are using the inner product, but you can simply change it to anything
    you want
    
    PreC: X1 and X2 have the same dimension
    """
    assert X.shape != Y.shape, "feature vectors should have the same dimension"
    return np.inner(X1,X2)

if __name__ == '__main__':
     # Read in the data
    train = np.genfromtxt('train_2cat.csv',delimiter = ',')
    """    
    X = train[:,1:94]
    Y = (train[:,94]).ravel()

    # Randomly pick 1000 data as the training data
    random.seed(100)
    sample = random.sample(range(len(Y)),1000)
    X = X[sample,]
    Y = Y[sample]
    """
    # Create a svm predictor
    C = 0.1
    tol = 0.001
    eps = 0.001
    svm = SVMTrainer(C, eps, tol)
    X = np.array([[-1,-1],[1,1]])
    Y = np.array([-1,1,])
    
    SMOtestObj = SMO(X,Y,C,tol,eps)
    SMOtestObj.SMO_main()
    predictor = svm.train(X,Y)
    
    
    """
    # Randomly pick 1000 data as the test data
    sample2 = random.sample(range(len(Y)),1000)
    test_X = X[sample2,]
    """
    
    # Make prediction
    #predict_Y = predictor.predict(test_X)
    
    
    