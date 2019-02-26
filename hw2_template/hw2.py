import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class XORNet(nn.Module):
    D_in=2
    H=2
    D_out=1 

    def __init__(self):#, optimizer=optim.sgd, loss_fn=):
        """
        Initialize the layers of your neural network

        You should use nn.Linear
        """
        super(XORNet, self).__init__()
        
        # self.loss_fn = loss_fn
        # self.optimizer = optimizer

        self.l1 = nn.Linear(self.D_in, self.H)
        self.l2 = nn.Linear(self.H, self.D_out)
    
    def set_l1(self, w, b):
        """
        Set the weights and bias of your first layer
        @param w: (2,2) torch tensor
        @param b: (2,) torch tensor
        """
        self.l1.weight = nn.Parameter(w)
        self.l1.bias = nn.Parameter(b)
    
    def set_l2(self, w, b):
        """
        Set the weights and bias of your second layer
        @param w: (1,2) torch tensor
        @param b: (1,) torch tensor
        """
        self.l2.weight = nn.Parameter(w)
        self.l2.bias = nn.Parameter(b) 
    
    def forward(self, xb):
        """
        Compute a forward pass in your network.  Note that the nonlinearity should be F.relu.
        @param xb: The (n, 2) torch tensor input to your model
        @return: an (n, 1) torch tensor
        """
        xb = self.l1(xb)
        xb = F.relu(xb)
        xb = self.l2(xb)

        return xb.view(xb.size())


    # def backward(self, X, y, o):

    #     self.optimizer.zero_grad()
    #     Y_pred = net(X)
    #     loss = self.loss_fn(Y_pred, Y)

    #     epoch_loss.append(loss)

    #     loss.backward()
    #     self.optimizer.step()

def fit(net, optimizer,  X, Y, n_epochs):
    """ Fit a net with BCEWithLogitsLoss.  Use the full batch size.
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param X: an (N, D) torch tensor
    @param Y: an (N, 1) torch tensor
    @param n_epochs: int, the number of epochs of training
    @return epoch_loss: Array of losses at the beginning and after each epoch. Ensure len(epoch_loss) == n_epochs+1
    """
    loss_fn = nn.BCEWithLogitsLoss() #note: input to loss function needs to be of shape (N, 1) and (N, 1)
    with torch.no_grad():
        epoch_loss = [loss_fn(net(X), Y)]
    for _ in range(n_epochs):

        #compute the loss for X, Y
        optimizer.zero_grad()
        Y_pred = net(X)
        loss = loss_fn(Y_pred, Y)

        #append the current loss to epoch_loss
        epoch_loss.append(loss)

        #it's gradient
        loss.backward()

        #optimize
        optimizer.step()


    return epoch_loss




class DigitsConvNet(nn.Module):
    def __init__(self):
        """ Initialize the layers of your neural network

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by 
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        """
        super(DigitsConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3)
        self.fc1 = nn.Linear(4, 10)

        

    def set_parameters(self, kern1, bias1, kern2, bias2, fc_weight, fc_bias):
        """ Set the parameters of your network

        @param kern1: an (8, 1, 3, 3) torch tensor
        @param bias1: an (8,) torch tensor
        @param kern2: an (4, 8, 3, 3) torch tensor
        @param bias2: an (4,) torch tensor
        @param fc_weight: an (10, 4) torch tensor
        @param fc_bias: an (10,) torch tensor
        """
        self.conv1.weight = nn.Parameter(kern1)
        self.conv1.bias = nn.Parameter(bias1) 
        self.conv2.weight = nn.Parameter(kern2)
        self.conv2.bias = nn.Parameter(bias2) 


        self.fc1.weight = nn.Parameter(fc_weight)
        self.fc1.bias = nn.Parameter(fc_bias) 



    def intermediate(self, xb):
        """ Return the feature representation your network lerans

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs. Hint: this
        should be very similar to your forward method
        @param xb: an (N, 8, 8) torch tensor
        @return: an (N, 4) torch tensor
        """


        x = F.relu(self.conv1(xb))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        return x

    def forward(self, xb):
        """ A forward pass of your neural network

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs
        @param xb: an (N, 8, 8) torch tensor
        @return: an (N, 10) torch tensor
        """
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        x = self.intermediate(xb)
        # x = x.view(-1, 4)
        x = self.fc1(x)


        return x.view(-1, 10)
        
def loss_batch(net, loss_func, X, Y):
    Y_pred = net(X)
    loss = loss_fn(Y_pred, Y)
    return loss



def fit_and_validate(net, optimizer, train, val, n_epochs, loss_func = nn.CrossEntropyLoss(), batch_size=1):
    """
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param train: a torch.utils.data.Dataset
    @param val: a torch.utils.data.Dataset
    @param n_epochs: the number of epochs over which to do gradient descent
    @param batch_size: the number of samples to use in each batch of gradient descent
    @return train_epoch_loss, validation_epoch_loss: two arrays of length n_epochs+1, containing the mean loss at the beginning of training and after each epoch
    """


    net.eval() #put the net in evaluation mode
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    val_dl = torch.utils.data.DataLoader(val)
    with torch.no_grad():
        # compute the mean loss on the training set at the beginning of iteration
        losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
        train_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        val_epoch_loss = []
        # TODO compute the validation loss and store it in a list


    for _ in range(n_epochs):
        net.train() #put the net in train mode
        # TODO 
        optimizer.zero_grad()

        losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
        train_epoch_loss += [np.sum(np.multiply(losses, nums)) / np.sum(nums)]

        for loss in losses:
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            # TODO compute the train and validation losses and store it in a list
            losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
            val_epoch_loss += [np.sum(np.multiply(losses, nums)) / np.sum(nums)]

    return train_epoch_loss, val_epoch_loss


def reconstruct_SVD(img, k, best=True):
    """ Compute the thin SVD for each channel of an image, keep only k singular values, and reconstruct a lossy image

    You should use numpy.linalg.svd, np.diag, and matrix multiplication
    @param img: a (M, N, 3) numpy ndarray 
    @param k: the number of singular value to keep
    @param best: Keep the k largest singular values if True.  Otherwise keep the k smallest singular values
    @return new_img: the (M, N, 3) reconstructed image
    """
    n = img.shape[0]

    new_img = np.zeros(img.shape)
    u_R, sigma_R, vt_R = np.linalg.svd(img[:,:,0])
    u_G, sigma_G, vt_G = np.linalg.svd(img[:,:,1])
    u_B, sigma_B, vt_B = np.linalg.svd(img[:,:,2])
    

    r = range(k) if best else range(n-1, n-k-1, -1)

    for i in r:

        new_img[:,:,0] += sigma_R[i] * np.outer(u_R[:, i], vt_R[i])
        new_img[:,:,1] += sigma_G[i] * np.outer(u_G[:, i], vt_G[i])
        new_img[:,:,2] += sigma_B[i] * np.outer(u_B[:, i], vt_B[i])
    
    return new_img
















