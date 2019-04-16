import hw2_utils as utils
import hw2 as hw
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 



def _2b():

    img = utils.get_image()
    u_R, sigma_R, vt_R = np.linalg.svd(img[:,:,0])
    plt.figure()
    plt.plot(np.log(sigma_R + 1.))
    plt.title('SVD for Red Color Channel')
    plt.ylabel('ln(1 + sigma)')
    plt.xlabel('i')
    plt.show()


def _2c():

    img = utils.get_image()
    params = list(zip([100, 50, 20, 10, 600], [1, 1, 1, 1, 0]))
    titles = [f'k = {k}, {"Max" if b else "Min"}' for k, b in params]
    imgs = [hw.reconstruct_SVD(img, k, best=b) for k, b in params]

    _, axs = plt.subplots(3, 2, figsize=(3, 2))
    axs = axs.flatten()
    for img, ax, tit in zip(imgs, axs, titles):
        ax.imshow(img)
        ax.set_title(tit)
        ax.set_axis_off()

    axs[-1].set_axis_off()
    plt.show()


def _3c():
    X, Y = utils.XOR_data()
    net = hw.XORNet()

    hw.fit(net, optim.SGD(net.parameters(), lr=.1),  X, Y, 5000)
    
    utils.contour_torch(X.min(), X.max(), Y.min(), Y.max(), net)


def _4_plotter(train_el, val_el, title):
    plt.figure()
    plt.plot(train_el, 'k--', label='Training Loss')
    plt.plot(val_el, 'k:', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropyLoss')
    plt.legend(shadow=True)
    plt.show()

def _4c():
    net = hw.DigitsConvNet()
    train, val = utils.torch_digits()
    n_epochs = 30
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    loss_func = nn.CrossEntropyLoss()

    train_el, val_el = hw.fit_and_validate(net, optimizer, loss_func, train, val, n_epochs)

    _4_plotter(train_el, val_el, 'SGD, lr = 0.005, batch = 1, epochs = 30')


def _4d():
    net = hw.DigitsConvNet()
    train, val = utils.torch_digits()
    n_epochs = 30
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    loss_func = nn.CrossEntropyLoss()

    train_el, val_el = hw.fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, sch=scheduler)
    torch.save(net.cpu().state_dict(), './conv.pb')

    _4_plotter(train_el, val_el, 'SGD, lr = 0.005, gamma = 0.95, batch = 1, epochs = 30')

def _4e():
    net = hw.DigitsConvNet()
    train, val = utils.torch_digits()
    n_epochs = 30
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    loss_func = nn.CrossEntropyLoss()

    train_el, val_el = hw.fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=16)

    _4_plotter(train_el, val_el, 'SGD, lr = 0.005, batch = 16, epochs = 30')

def _4f():
    net = hw.DigitsConvNet()
    train, val = utils.torch_digits()
    n_epochs = 30
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    loss_func = nn.CrossEntropyLoss()

    train_el, val_el = hw.fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=16, sch=scheduler)
    digits, labels, val_X, val_Y = utils.torch_digits(mode=True)

    intermediate = net.intermediate(digits)

    utils.plot_PCA(intermediate.cpu().detach().numpy(), labels.cpu().detach().numpy())

def _4g():
    net = hw.DigitsConvNet()
    train, val = utils.torch_digits()
    n_epochs = 30
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    loss_func = nn.CrossEntropyLoss()

    train_el, val_el = hw.fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=16, sch=scheduler)

    digits, labels, val_X, val_Y = utils.torch_digits(mode=True)

    labels = labels.cpu().detach().numpy()
    val_Y = val_Y.cpu().detach().numpy()
    intermediate_train = net.intermediate(digits).cpu().detach().numpy()
    intermediate_val = net.intermediate(val_X).cpu().detach().numpy()

    knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    knn.fit(intermediate_train, labels)
    accuracy = knn.score(intermediate_val, val_Y) 

    print(accuracy)

    # tree = KDTree(np.array([(int_np[i], l_np[i]) for i in range(n)]))
    # tree.query()


    # losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y, opt=optimizer) for X, Y in train_dl])
    # err = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]



if __name__ == '__main__':
    # _2b()
    # _2c()

    _3c()

    # _4c()
    # _4d()
    # _4e()
    # _4f()
    # _4g()




