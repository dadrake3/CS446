import numpy as np
import hw1_utils as utils
import torch 
from matplotlib import pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
# Problem 2
def gd(X,y,loss,lrate,num_iter):
    w = torch.zeros(X.shape[1], requires_grad=True)
    for i in range(num_iter):
        l = loss(X,y,w).mean() 
        # print(l)
        l.backward()

        with torch.no_grad():
            w -= lrate * w.grad
            w.grad.zero_()

    return w

least_squares_loss = lambda X,y,w: ((torch.matmul(X, w) - y) ** 2) * 0.5

def linear_gd(X,y,lrate=0.1,num_iter=1000):

    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    X = torch.tensor(X, requires_grad=True).type(torch.FloatTensor)
    y = torch.tensor(y, requires_grad=True).type(torch.FloatTensor)

    w = gd(X,y,least_squares_loss,lrate,num_iter)

    w_numpy = w.detach().numpy()
   
    return w_numpy





def linear_normal(X,Y):
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    # return parameters as numpy array
    A = (X.shape[0] ** -0.5) * X
    b = (X.shape[0] ** -0.5) * Y

    w = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, b))

    return w

def plot_linear():
    X,y = utils.load_reg_data()

    w_numpy = linear_normal(X,y)

    X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
    X = torch.tensor(X, requires_grad=True).type(torch.FloatTensor)
    y = torch.tensor(y, requires_grad=True).type(torch.FloatTensor)
    # return plot
    X_numpy = X.detach().numpy()[:,0]
    y_numpy = y.detach().numpy()

    # utils.contour_plot(min(X_numpy), max(X_numpy), min(y_numpy), max(y_numpy), M, ngrid = 33)

    plt.figure()
    plt.plot(X_numpy,y_numpy)
    plt.plot(X_numpy, X_numpy * w_numpy[1] + w_numpy[0])
    plt.title('Linear Normal Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Problem 4
def expand_x(X):
        X_expand = []
        # print(X)
        for row in range(X.shape[0]):
            x_deg_0 = [1]
            x_deg_1 = list(X[row,:])
            x_deg_2 = [X[row,i] * X[row,j] for i in range(X.shape[1]) for j in range(X.shape[1]) if j >= i]

            X_expand.append(x_deg_0 + x_deg_1 + x_deg_2)
            
        return np.array(X_expand)

def poly_gd(X,y,lrate=0.01,num_iter=3000):
    # return parameters as numpy array
    

    X = torch.tensor(expand_x(X), requires_grad=True).type(torch.FloatTensor)
    y = torch.tensor(y, requires_grad=True).type(torch.FloatTensor)

    w = gd(X,y,least_squares_loss,lrate,num_iter)

    w_numpy = w.detach().numpy()
   
    return w_numpy


def poly_normal(X,Y):
    # return parameters as numpy array

    # X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    X = expand_x(X)
    # return parameters as numpy array
    A = (X.shape[0] ** -0.5) * X
    b = (X.shape[0] ** -0.5) * Y

    w = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, b))

    return w

def plot_poly():
    # return plot
    X, Y = utils.load_reg_data()
    w = poly_normal(X, Y)

    plt.plot(X, Y, X, (w[2]* X**2 + w[1]*X + w[0]))

    plt.title('Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


def poly_xor():
    # return labels for XOR from linear,polynomal models




    y_poly = []
    y_linear = []
    return y_linear,y_poly


# Problem 5
def nn(X,y,X_test):
    # return labels for X_test as numpy array
    X = X.astype(float)
    y = y.astype(float)
    X_test = X_test.astype(float)

    # y_pred = np.array([float('inf')]*X_test.shape[0])
    y_pred = np.zeros(X_test.shape[0]).astype(float)
    for i in range(X_test.shape[0]):

        curr_min = float('inf')
        curr_label = 0.
        for j in range(X.shape[0]):
            # dist = np.linalg.norm(X[j,:] - X_test[i,:])
            # dist = ((X[j,:] - X_test[i,:]) ** 2.).sum() ** 0.5


            curr_sum = 0.
            for k in range(X.shape[1]):
                curr_sum += (X[j,k] - X_test[i,k]) ** 2.
            dist = curr_sum ** 0.5

            curr_label = y[i] if dist < curr_min else curr_label
            curr_min = dist if dist < curr_min else curr_min

        y_pred[i] = curr_label

    return y_pred


# split into to test and train
def nn_iris():
    X,y = utils.load_reg_data()
    n = X.shape[0]
    X_test = X[:int(n*0.3),:]
    y_test = y[:int(n*0.3)]

    # model
    X_train = X[int(n*0.3)+1:,:]
    y_train = y[int(n*0.3)+1:]

    y_hat = nn(X_train, y_train, X_test)

    print(y_hat)
    print(y_test)

    error = np.count_nonzero(np.abs(y_hat - y_test)) / y_test.shape[0]

    return error


# Problem 6
# logistic_loss = lambda X,y,w: torch.tensor([torch.log(1 + torch.exp(-y[i] * torch.matmul(w, X[i,:]))) for i in range(X.shape[0])], requires_grad=True)
logistic_loss = lambda X, y, w: torch.log (1 + torch.exp(-y * torch.matmul(X, w)))

def logistic(X,y,lrate=1,num_iter=3000):

    X = torch.tensor(X, requires_grad=True).type(torch.FloatTensor)
    y = torch.tensor(y, requires_grad=True).type(torch.FloatTensor)

    w = gd(X,y,logistic_loss,lrate,num_iter)

    w_numpy = w.detach().numpy()

    return w_numpy

def logistic_vs_ols():
    # return plot
    # X,y = utils.load_logistic_data()

    # w_log = logistic(X,y)
    # w_lin = linear_gd(X,y)


    # # utils.contour_plot(min(X_numpy), max(X_numpy), min(y_numpy), max(y_numpy), M, ngrid = 33)


    
    # # plt.plot(X, X * w_lin[1] + w_lin[0])
    # plt.clf()
    # plt.title('Linear Normal Regression')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    X, Y = utils.load_logistic_data()
    print(X.shape, Y.shape)

    w_lin = linear_normal(X, Y)
    print(w_lin.shape)
    # plt.plot(X,Y)
    # plt.plot(X, X * w_lin[1] + w_lin[0])


    # plt.title('Polynomial Regression')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y)
    ax.scatter(X[:,0], X[:,1], X[:,0] * w_lin[0] + w_lin[1] * X[:,1] - w_lin[0])


    # y_hat = X[:,0] * w_lin[1] + w_lin[0] + 
    # ax.scatter(X[:,0], X[:,0], y_hat)

    plt.show()





    return []


if __name__ == '__main__':
    # X,y = utils.load_logistic_data()
    # X = np.hstack((X, 3*X, .1*X))
    # plot_linear()
    # X, y = utils.load_nn_data()
    # utils.voronoi_plot(X,y)
    # print(logistic(X,y))

    # nn_iris()
    
    # plot_poly()
    logistic_vs_ols()








