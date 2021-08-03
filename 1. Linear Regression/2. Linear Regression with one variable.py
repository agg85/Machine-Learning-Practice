import os								#for manipulating directory paths
import numpy as np						#vectors and arrays
from matplotlib import pyplot 			#plotting lib
from mpl_toolkits.mplot3d import Axes3D #for plotting 3D surfaces

#assignment submission and grading related
import utils
grader = utils.Grader()

# %matplotlib inline

################# Linear Regression with one variable ##################### 

#Loading data:

data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:,0], data[:,1]

#no. of training examples
m = y.size

############ (2.1) Plotting the data ############

def plotData(x,y):
	# x and y should have the same size
	fig = pyplot.figure()
	pyplot.plot(x,y,'ro',ms=10,mec='k')
	pyplot.ylabel('Profit in 10000s')
	pyplot.xlabel('Population in City in 10000s')
	# pyplot.show()

plotData(X,y)

############ (2.2) Gradient Descent ############

# axis=1 means columns
# axis=0 means rows
X = np.stack([np.ones(m),X], axis=1)
# X is now (m x n+1): m- no. of training examples and n- no. of features

def computeCost(X,y,theta):
	# X: (m x n+1), theta: (n+1 x 1), y: (m x 1)
	# (m x n+1) x (n+1 x 1) = (m x 1)
	J = np.sum(np.power((X.dot(theta)-y),2))/(2*y.size)
	return J

J = computeCost(X,y,theta=np.array([0.0,0.0]))
print('With theta = [0,0] initially, \n%.2f is the cost computed' % J)
print('Expected cost value is 32.07\n')
J = computeCost(X,y,theta=np.array([-1,2]))
print('With theta = [-1,2] initially, \n%.2f is the cost computed' % J)
print('Expected cost value is 54.24\n')

grader[2] = computeCost
# grader.grade()

def gradientDescent(X, y, theta, alpha, num_iters):
	# X: (m x n+1), y: (m, ), theta: (n+1, )
	m = X.shape[0]
	theta = theta.copy()
	J_history = []
	for i in range(num_iters):
		# # theta is:			     	(n+1 x 1)
		# # np.dot(X,theta) and y are: 	(m x 1)
		# # X is:   			  		(m x n+1)
		# # X.T is:             		(n+1 x m)
		theta -= (alpha/m)*np.dot(X.T, np.dot(X,theta)-y)
		theta.reshape(X.shape[1],1)
		J_history.append(computeCost(X, y, theta))
	return theta, J_history

theta = np.zeros(2)
iterations = 1500
alpha = 0.01
theta, J_history = gradientDescent(X,y,theta,alpha,iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

plotData(X[:,1],y)
pyplot.plot(X[:,1],np.dot(X,theta),'-')
pyplot.legend(['Training data','Linear regression'])
pyplot.show()

predict1 = np.dot([1,3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n' .format(predict1*10000))
predict2 = np.dot([1,7.0], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n' .format(predict2*10000))

grader[3] = gradientDescent
grader.grade()