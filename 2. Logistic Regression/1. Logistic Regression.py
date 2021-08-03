import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize

# For submission
import utils
grader = utils.Grader()

############ (1) Logistic Regression ############

###### (1.1) Data Loading ######

data = np.loadtxt(os.path.join('Data','ex2data1.txt'),delimiter=',')
X = data[:,:2]
y = data[:, 2]

def plotData(X,y):
	fig = pyplot.figure()
	pos = y==1
	neg = y==0
	pyplot.plot(X[pos,0],X[pos,1],'k*',lw=2,ms=10)
	pyplot.plot(X[neg,0],X[neg,1],'ko',mfc='y',ms=8,mec='k',mew=1)

plotData(X,y)
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not Admitted'])
# pyplot.show()
pass

######### (1.2) Implementation #########

# y = np.reshape(y,(y.size,1))

###### (1.2.1) WarmUp Exercise ######

def sigmoid(z):
	z = np.array(z)
	g = np.zeros(z.shape)
	g = 1/(1+np.exp(-z))
	return g

z = 0
g = sigmoid(z)
print(g)

grader[1] = sigmoid
# grader.grade()

###### (1.2.2) Cost function and gradient ######

# print(X.shape)
X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
# print(X.shape)

def costFunction(theta, X, y):
	J = 0
	grad = np.zeros(theta.shape)
	hypo = sigmoid(X.dot(theta))
	m = X.shape[0]
	J = (y.T.dot(np.log(hypo)) + (1-y).T.dot(np.log(1-hypo)))*(-1/m)
	grad = (1/m)*(X.T.dot(hypo-y))
	return J, grad

initial_theta = np.zeros(X.shape[1])
cost, grad = costFunction(initial_theta, X, y)
print('Expected cost is 0.693')
print('Cost computed: ', cost, '\n')
print('Expected gradient: [-0.1000, -12.0092, -11.2628]')
print('Gradient: ', grad, '\n')

test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Expected cost is 0.218')
print('Cost computed: ', cost, '\n')
print('Expected gradient: [0.043, 2.566, 2.647]')
print('Gradient: ', grad, '\n')

grader[2] = costFunction
grader[3] = costFunction
# grader.grade()

###### (1.2.3) Gradient Descent using scipy's optimize.minimize ######

# scipy.optimize.minimize in Python is alternative for fminunc in MATLAB
options = {'maxiter': 400}
res = optimize.minimize(costFunction,
						initial_theta,
						(X,y),
						jac = True,
						method = 'TNC',
						options = options)

# 'fun' property of res returns value of costFunction at optimized theta
# optimized theta is in 'x' property of res
cost = res.fun
theta = res.x

print('Expected cost (approx): 0.203');
print('Cost at theta found by optimize.minimize: ', cost, '\n')

print('Expected theta (approx): [-25.161, 0.206, 0.201]')
print('theta: ', theta, '\n')

utils.plotDecisionBoundary(plotData, theta, X, y)
# pyplot.show()

###### (1.2.4) Evaluating Logistic Regression ######

def predict(theta, X):
	p = np.zeros(X.shape[0])
	# theta = np.reshape(theta,(theta.size,1))
	p = np.round(sigmoid(X.dot(theta)))
	return p

prob = sigmoid(np.dot([1,45,85],theta))

print('Expected value: 0.775 +/- 0.002')
print('The predicted probability: ', prob, '\n')

p = predict(theta, X)
print('Expected accuracy: 89.00%')
print('Train accuracy: ', np.mean(p==y)*100)

grader[4] = predict
grader.grade()