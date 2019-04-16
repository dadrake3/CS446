import hw3_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.svm import SVC  

def hinge_loss(y_pred, y_true):
    return torch.mean(torch.clamp(1 - y_pred.t() * y_true, min=0))

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    """An SVM solver.

    Arguments:
        x_train: a 2d tensor with shape (n, d).
        y_train: a 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: the learning rate.
        num_iters: the number of gradient descent steps.
        kernel: the kernel function.
           The default kernel function is 1 + <x, y>.
        c: the trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Return:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step. 
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    """
    a = torch.zeros(x_train.shape[0])

    for j in range(0, num_iters):

        X = [[kernel(x_train[i], x) for x in x_train] for i in range(x_train.shape[0])]
        grad = torch.stack([y_train[i] * torch.sum(a * y_train @ torch.tensor(X)) - 1 for i in range(x_train.shape[0])])

        a = torch.clamp_(a - (lr * grad), 0., c)

    return a




def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    """An SVM predictor.

    Arguments:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: a 2d tensor with shape (n, d), denoting the training set.
        y_train: a 1d tensor with shape (n,), whose elememnts are +1 or -1.
        x_test: a 2d tensor with shape (m, d), denoting the test set.
        kernel: the kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    """

    y_hat = torch.zeros(x_test.shape[0])
    for j in range(x_test.shape[0]):
        y_hat[j] = sum([alpha[i] * y_train[i] * kernel(x_train[i], x_test[j]) for i in range(x_train.shape[0])])

    return y_hat

  

def svm_contour(alpha, x_train, y_train, kernel,
                xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33):
    """Plot the contour lines of the svm predictor. """
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)),
            dim = 2).view(-1, 2)
        zz = svm_predictor(alpha, x_train, y_train, x_test, kernel)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                        cmap = 'RdYlBu')
        plt.clabel(cs)
        plt.show()


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()


        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(num_channels)

        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(num_channels)




    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
       
        l1 = self.conv1(x)
        l1 = self.bnorm1(l1)
        l1 = F.relu(l1)

        l2 = self.conv2(l1)
        l2 = self.bnorm2(l2)

        y = F.relu(x + l2)

        return y

    def set_param(self, kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_1: a (C, C, 3, 3) tensor, kernels of the first conv layer.
            bn1_weight: a (C,) tensor.
            bn1_bias: a (C,) tensor.
            kernel_2: a (C, C, 3, 3) tensor, kernels of the second conv layer.
            bn2_weight: a (C,) tensor.
            bn2_bias: a (C,) tensor.
        """
        self.conv1.weight = nn.Parameter(kernel_1)

        self.bnorm1.weight = nn.Parameter(bn1_weight)
        self.bnorm1.bias = nn.Parameter(bn1_bias) 

        self.conv2.weight = nn.Parameter(kernel_2)

        self.bnorm2.weight = nn.Parameter(bn2_weight)
        self.bnorm2.bias = nn.Parameter(bn2_bias) 

        
        


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()



        self.conv0 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.bnorm0 = nn.BatchNorm2d(num_channels)
        self.mpool = nn.MaxPool2d(2)
        self.block = Block(num_channels)
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)





       

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """


        l1 = self.conv0(x)
        l1 = self.bnorm0(l1)
        l1 = F.relu(l1)
        l1 = self.mpool(l1)

        l2 = self.block(l1)
        l2 = self.apool(l2)

        l3 = l2.view(l2.shape[0], -1)
        y = self.fc(l3)

        return y


    def set_param(self, kernel_0, bn0_weight, bn0_bias,
                  kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias,
                  fc_weight, fc_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_0: a (C, 1, 3, 3) tensor, kernels of the conv layer
                      before the building block.
            bn0_weight: a (C,) tensor, weight of the batch norm layer
                        before the building block.
            bn0_bias: a (C,) tensor, bias of the batch norm layer
                      before the building block.
            fc_weight: a (10, C) tensor
            fc_bias: a (10,) tensor
        See the docstring of Block.set_param() for the description
        of other arguments.
        """
        self.conv0.weight = nn.Parameter(kernel_0)
        self.bnorm0.weight = nn.Parameter(bn0_weight)
        self.bnorm0.bias = nn.Parameter(bn0_bias) 

        self.block.set_param(kernel_1, bn1_weight, bn1_bias, kernel_2, bn2_weight, bn2_bias)

        self.fc.weight = nn.Parameter(fc_weight)
        self.fc.bias = nn.Parameter(fc_bias)














