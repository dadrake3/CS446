import hw2_utils as utils
import hw2 as hw
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt 

def _2b():

    img = utils.get_image()
    u_R, sigma_R, vt_R = np.linalg.svd(img[:,:,0])
    plt.figure()
    plt.plot(np.log(sigma_R + 1.))
    plt.title('ln(1 + sigma) for red channel')

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

    fit(net, optim.SGD(net.parameters(), lr=.1),  X, Y, 5000)
    
    utils.contour_torch(X.min(), X.max(), Y.min(), Y.max(), net)
    # hw.contour_torch(-1., 1., -1., 1., net)




# def _4c():
#     (train, val) = hw.torch_digits()
#     net = DigitsConvNet()
#     (train_el, val_el) = fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=1)



if __name__ == '__main__':
    _2b()
    # _2c()

    # _3c()
