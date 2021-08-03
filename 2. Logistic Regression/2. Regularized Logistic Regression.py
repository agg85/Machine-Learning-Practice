import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize

# For submission
import utils
grader = utils.Grader()

############ (2) Regularised Logistic Regression ############

###### (2.1) Data Loading and Visualisation ######
data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:,:2]
y = data[:, 2]
# print(data)
# print(X)
# print(y)

def plotData(X,y):
	fig = pyplot.figure()
	pos = y==1
	neg = y==0
	pyplot.plot(X[pos,0],X[pos,1],'k*',lw=2,ms=10)
	pyplot.plot(X[neg,0],X[neg,1],'ko',mfc='y',ms=8,mec='k',mew=1)

# plotData(X, y)
# pyplot.xlabel('Microchip Test 1')
# pyplot.ylabel('Microchip Test 2')
# pyplot.legend(['y = 1', 'y = 0'], loc='upper right')
# pyplot.show()
# pass

def sigmoid(z):
	z = np.array(z)
	g = np.zeros(z.shape)
	g = 1/(1+np.exp(-z))
	return g

###### (2.2) Feature Mapping ######
X = utils.mapFeature(X[:, 0], X[:, 1])

###### (2.3) Cost Function and gradient ######

def costFunctionReg(theta, X, y, lambda_):
	J = 0
	grad = np.zeros(theta.shape)
	hypo = sigmoid(X.dot(theta))
	m = X.shape[0]
	theta = theta.copy()
	theta[0] = 0
	J = (y.T.dot(np.log(hypo)) + (1-y).T.dot(np.log(1-hypo)))*(-1/m) + (lambda_/(2*m))*(np.sum(np.power(theta,2)))
	grad = (1/m)*(X.T.dot(hypo-y)) + (lambda_/m)*theta
	return J, grad

initial_theta = np.zeros(X.shape[1])
lambda_ = 1
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

# print('Cost at test theta    : {:.2f}'.format(cost))
# print('Expected cost (approx): 3.16\n')

# print('Gradient at test theta - first five values only:')
# print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
# print('Expected gradients (approx) - first five values only:')
# print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]\n')

grader[5] = costFunctionReg
grader[6] = costFunctionReg
# grader.grade()

###### (2.4, 2.5) Set regularization Parameter (lambda_) ######

initial_theta = np.zeros(X.shape[1])
lambda_ = 1
res = optimize.minimize(costFunctionReg,
						initial_theta,
						(X,y,lambda_),
						jac = True,
						method = 'TNC',
						options = {'maxiter': 100})

cost = res.fun
theta = res.x

utils.plotDecisionBoundary(plotData, theta, X, y)
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
pyplot.legend(['y = 1', 'y = 0'])
pyplot.grid(False)
pyplot.title('lambda = %0.2f' % lambda_)
pyplot.show()

def predict(theta, X):
	return np.round(sigmoid(X.dot(theta)))

p = predict(theta, X)

print('Expected accuracy (with lambda = 1): 83.1 % (approx)')
print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))