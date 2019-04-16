import hw3_utils as utils
import hw3 as hw
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt 


def fit_and_validate(model, optimizer, criterion, train, val, n_epochs, batch_size=1):
    """
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param train: a torch.utils.data.Dataset
    @param val: a torch.utils.data.Dataset
    @param n_epochs: the number of epochs over which to do gradient descent
    @param batch_size: the number of samples to use in each batch of gradient descent
    @return train_epoch_loss, validation_epoch_loss: two arrays of length n_epochs+1, containing the mean loss at the beginning of training and after each epoch
    """
    # print(type(loss_func))
    model.eval() #put the net in evaluation mode
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    val_dl = torch.utils.data.DataLoader(val)
    # with torch.no_grad():   
    #     # compute the mean loss on the training set at the beginning of iteration
    #     losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
    #     train_el = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
    #     # TODO compute the validation loss and store it in a list
    #     losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
    #     val_el = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
    train_el = []
    val_el = []

    for _ in range(n_epochs):
        # print(val_el)
        model.train() #put the net in train mode
        # TODO
        print(_)

        epoch_loss = []

        for _, (inputs, labels) in enumerate(train_dl):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss)

        train_el.append(sum(epoch_loss) / len(epoch_loss))
        

        # losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y, opt=optimizer) for X, Y in train_dl])
        # train_el += [np.sum(np.multiply(losses, nums)) / np.sum(nums)]

        epoch_loss = []

        for _, (inputs, labels) in enumerate(val_dl):
            with torch.no_grad():
                model.eval() #put the net in evaluation mode
                # TODO compute the train and validation losses and store it in a list

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                epoch_loss.append(loss)

        val_el.append(sum(epoch_loss) / len(epoch_loss))

                

    return train_el, val_el 




def plotter(train_el, val_el, title):
    plt.figure()
    plt.plot(train_el, label='Training Loss')
    plt.plot(val_el, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropyLoss')
    plt.legend(shadow=True)
    plt.show()

def _3d(title):
    model = hw.ResNet(16)
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    loss_func = nn.CrossEntropyLoss()
    n_epochs = 30
    batch_size = 16

    train, val = utils.torch_digits()
    train_l, val_l = fit_and_validate(model, optimizer, loss_func, train, val, n_epochs, batch_size=1)
    plotter(train_l, val_l, title)


def _6_c():
    x_train, y_train = utils.xor_data()

    # kernel=utils.poly(degree=1)
    kernel = utils.rbf(4)
    alpha = hw.svm_solver(x_train, y_train, 0.1, 1000,
               kernel=kernel, c=None)



    hw.svm_contour(alpha, x_train, y_train, kernel,
                xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33)

if __name__ == '__main__':
    _3d('Resnet trial 1')
    # _3d('Resnet trial 2')
    # _3d('Resnet trial 3')
    # _6_c()




















