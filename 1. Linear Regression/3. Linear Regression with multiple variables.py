import os								#for manipulating directory paths
import numpy as np						#vectors and arrays
from matplotlib import pyplot 			#plotting lib
from mpl_toolkits.mplot3d import Axes3D #for plotting 3D surfaces

#assignment submission and grading related
import utils
grader = utils.Grader()

# %matplotlib inline

################# Linear Regression with multiple variables #####################

###### (3.1) Feature Normalization ######

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:,:2]
y = data[:, 2]
y = np.reshape(y,(y.size,1))

# print('{:>8s}{:>8s}{:>15s}'.format('X[:,0]', 'X[:,1]', 'y'))
# print('-'*26)
# for i in range(10):
# 	print('{:>8s}{:>8s}{:>15s}'.format(str(X[i, 0]), str(X[i, 1]), str(y[i])))

def featureNormalize(X):
	X_norm = X.copy()
	mu = np.zeros(X.shape[1])
	sigma = np.zeros(X.shape[1])
	mu = np.mean(X_norm, axis=0)
	sigma = np.std(X_norm, axis=0)
	X_norm = (X_norm-mu)/sigma
	return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

grader[4] = featureNormalize

# X = np.concatenate([np.ones((X.shape[0],1)), X_norm], axis = 1)
# X = np.concatenate([np.ones((X.shape[0],1)), X], axis = 1)

###### Compute cost ######

def computeCostMulti(X,y,theta):
	return np.sum(np.power(np.dot(X,theta)-y,2))/(2*X.shape[0])

grader[5] = computeCostMulti

###### Gradient Descent ######

def gradientDescentMulti(X, y, theta, alpha, num_iters):
	theta = theta.copy()
	J_history = []
	m = X.shape[0]
	for i in range(num_iters):
		theta -= (alpha/m)*np.dot(X.T, np.dot(X,theta)-y)
		J_history.append(computeCostMulti(X,y,theta))
	return theta, J_history

grader[6] = gradientDescentMulti

###### Alpha selection ######

alpha = 0.00000001
num_iters = 400
theta = np.zeros((X.shape[1],1))
theta, J_history = gradientDescentMulti(X,y,theta,alpha,num_iters)

pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.show()
print(theta)

sx = [1650,3]
sx = (sx-mu)/sigma
print(np.dot([[1, sx[0], sx[1]]], theta))

###### (3.3) Normal Eqn ######

def normalEqn(X, y):
	return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), y)
grader[7] = normalEqn

theta = normalEqn(X,y)
print(theta)
print(np.dot([1,1650,3],theta))