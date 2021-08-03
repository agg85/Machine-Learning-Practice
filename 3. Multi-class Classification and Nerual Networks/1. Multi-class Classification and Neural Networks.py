import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils
grader = utils.Grader()

############ (1) Multi-class Classification ############

def sigmoid(z):
    return 1/(1+np.exp(-z))

###### (1.1, 1.2) Loading and Visualising Data ######

input_layer_size = 400
# num_labels is the number of classes (K)
num_labels = 10                                         # we've mapped 0 to label 10 
data = loadmat(os.path.join('Data','ex3data1.mat'))
X,y = data['X'], data['y'].ravel()                      # ravel converts (n x m) 2D matrix to a (n*m x 1) linear vector
y[y==10] = 0                                            # mapping 0 to label 0
m = y.size

rand_indices = np.random.choice(y.size, 100, replace = False)
sel = X[rand_indices, :]
utils.displayData(sel)

###### (1.3) Vectorising Logistic Regression ######

# some testing values and parameters:
theta_t = np.array([-2, -1, 1, 2], dtype = float)
X_t = np.concatenate([np.ones((5,1)), np.arange(1,16).reshape(5, 3, order='F')/10.0], axis = 1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3

# vectorising cost function
def lrCostFunction(theta, X, y, lambda_):
    m = y.size
    if y.dtype == bool:
        y = y.astype(int)

    J = 0
    grad = np.array(theta.shape)

    h = sigmoid(X.dot(theta))
    theta = theta.copy()
    theta[0] = 0
    J = (-1/m)*(y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h))) + (lambda_/(2*m))*np.sum(np.power(theta,2))
    grad = (1/m)*(X.T.dot(h-y)) + (lambda_/m)*(theta)
    return J, grad

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

# print('Cost         : {:.6f}'.format(J))
# print('Expected cost: 2.534819')
# print('Gradients: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
# print('Expected gradients: [0.146561, -0.548558, 0.724722, 1.398003]')

grader[1] = lrCostFunction
# grader.grade()

###### (1.4) One-vs-all Classification

def oneVsAll(X, y, num_labels, lambda_):
    # all_theta is (k x n+1)
    # X is (m x n+1)
    m, n = X.shape
    all_theta = np.zeros((num_labels, n+1))
    X = np.concatenate([np.ones((m,1)), X], axis=1)

    for c in range(num_labels):
        initial_theta = np.zeros(n+1)
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X, y==c, lambda_),
                                jac = True,
                                method = 'CG',
                                options = {'maxiter': 50})
        all_theta[c] = res.x

    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

grader[2] = oneVsAll
# grader.grade()

def predictOneVsAll(all_theta, X):
    num_labels = all_theta.shape[0]
    m, n = X.shape
    X = np.concatenate([np.ones((m,1)), X], axis = 1)

    p = np.zeros(m)
    # argmax returns the index of row/col which has the max value in the col/row
    # axis = 0: row having max value in col
    # axis = 1: col having max value in row
    p = np.argmax(sigmoid(X.dot(all_theta.T)), axis = 1)
    return p

pred = predictOneVsAll(all_theta, X)
# print(pred.size)
# print(y.size)
# print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))

grader[3] = predictOneVsAll
# grader.grade()

############ (2) Nerual Networks ############

###### Loading Dataset ###### (Reloading to ensure nothing changes)

input_layer_size = 400
# num_labels is the number of classes (K)
num_labels = 10                                         # we've mapped 0 to label 10 
data = loadmat(os.path.join('Data','ex3data1.mat'))
X,y = data['X'], data['y'].ravel()                      # ravel converts (n x m) 2D matrix to a (n*m x 1) linear vector
y[y==10] = 0                                            # mapping 0 to label 0
m = y.size

indices = np.random.permutation(m)
rand_indices = np.random.choice(y.size, 100, replace = False)
sel = X[rand_indices, :]
utils.displayData(sel)

###### (2.1) Model Representation ######

input_layer_size = 400
hidden_layer = 25
num_labels = 10
weights = loadmat(os.path.join('Data','ex3weights.mat'))
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
#swapping first and last columns of Theta2 bcz of MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

###### (2.2) Feedforward Propagation and Prediction ######

def predict(Theta1, Theta2, X):
    # Theta 1 is (hidden_layer x input_layer+1)
    # Theta 2 is (output_layer x hidden_layer+1)
    # X is (m x n)
    # p is (m x 1)

    if X.ndim == 1:
        X = X[None] #np.reshape(X,(1,X.size))

    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros(X.shape[0])

    X = np.concatenate([np.ones((m,1)),X],axis=1)
    
    # Theta1: (h x n+1)
    # Theta2: (k x h+1)
    # X:      (m x n+1)
    # p:      (m x 1)
    # a2:     (m x h+1)

    a2 = np.concatenate([np.ones((m,1)),sigmoid(X.dot(Theta1.T))],axis=1)
    p = np.argmax(sigmoid(a2.dot(Theta2.T)),axis=1)
    return p

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred==y)*100))

while indices.size > 4000:
    print(indices.shape)
    i, indices = indices[0], indices[1:]
    print(i)
    print(X.shape)
    utils.displayData(X[i, :], figsize=(4,4))
    print(*predict(Theta1, Theta2, X[i, :]))

pyplot.show()

grader[4] = predict
grader.grade()