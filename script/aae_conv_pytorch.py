import argparse
import time
import torch
import pickle
import numpy as np
import itertools
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.nn.modules.upsampling import UpsamplingNearest2d

# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

seed = 10
if args.cuda:
    torch.cuda.manual_seed(seed)
cuda = args.cuda

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 10
z_dim = 10
X_dim = 784
y_dim = 10
convolutional = False
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
N = 1000
epochs = args.epochs


##################################
# Weights initialization
##################################
# From https://github.com/alykhantejani/nninit/blob/master/nninit.py
def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = np.prod(tensor.numpy().shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def kaiming_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Delving deep into rectifiers:
       Surpassing human-level performance on ImageNet classification" - He, K. et al using a normal distribution.
       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(1/(fan_in))
    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.kaiming_normal(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        kaiming_normal(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(1.0 / fan_in)
        return tensor.normal_(0, std)

##################################
# Load data and create Data loaders
##################################

print('loading data!')
data_path = './../data/'
trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
# Set -1 as labels for unlabeled data
trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * 47000))
validset = pickle.load(open(data_path + "validation.p", "rb"))

train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, **kwargs)

train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                     batch_size=train_batch_size,
                                                     shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)

##################################
# Define Networks
##################################
# Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        kaiming_normal(self.lin1.weight)
        self.lin2 = nn.Linear(N, N)
        kaiming_normal(self.lin2.weight)

        # Categorical code (y)
        self.lin3cat = nn.Linear(N, n_classes)
        kaiming_normal(self.lin3cat.weight)

        # Gaussian code (z)
        self.lin3gauss = nn.Linear(N, z_dim)
        kaiming_normal(self.lin3gauss.weight)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)

        xcat = F.softmax(self.lin3cat(x))
        xgauss = self.lin3gauss(x)

        return xcat, xgauss

# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim + n_classes, N)
        kaiming_normal(self.lin1.weight)
        self.lin2 = nn.Linear(N, N)
        kaiming_normal(self.lin2.weight)
        self.lin3 = nn.Linear(N, X_dim)
        kaiming_normal(self.lin3.weight)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator networks
class D_net_cat(nn.Module):
    def __init__(self):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(n_classes, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)


class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)


####################
# Utility functions
####################
def sample_categorical(batch_size, n_classes=10):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, 10, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)


def report_loss(epoch, D_loss_cat, D_loss_gauss, G_loss, recon_loss):
    '''
    Print loss
    '''
    print('Epoch-{}; D_loss_cat: {:.4}: D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch,
                                                                                                      D_loss_cat.data[0],
                                                                                                      D_loss_gauss.data[0],
                                                                                                      G_loss.data[0],
                                                                                                      recon_loss.data[0]))


def create_latent(Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        # X.resize_(loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
        z_sample = Q(X)
        if batch_idx > 0:
            z_values = np.concatenate((z_values, np.array(z_sample.data.tolist())))
        else:
            z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels)

    return z_values, labels


def classification_accuracy(Q, data_loader):
    Q.eval()
    labels = []
    test_loss = 0
    correct = 0

    for batch_idx, (X, target) in enumerate(data_loader):
        X = X * 0.3081 + 0.1307
        X.resize_(data_loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        labels.extend(target.data.tolist())
        # Reconstruction phase
        output = Q(X)[0]

        test_loss += F.nll_loss(output, target).data[0]

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader)
    return 100. * correct / len(data_loader.dataset)


####################
# Train procedure
####################
def train(P, Q, D_cat, D_gauss, P_decoder, Q_encoder, Q_semi_supervised, Q_generator, D_cat_solver, D_gauss_solver, train_labeled_loader, train_unlabeled_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()
    D_cat.train()
    D_gauss.train()

    if train_unlabeled_loader is None:
        train_unlabeled_loader = train_labeled_loader

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for (X_l, target_l), (X_u, target_u) in itertools.izip(train_labeled_loader, train_unlabeled_loader):

        for X, target in [(X_u, target_u), (X_l, target_l)]:
            if target[0] == -1:
                labeled = False
            else:
                labeled = True

            # Load batch and normalize samples to be between 0 and 1
            X = X * 0.3081 + 0.1307
            X.resize_(train_batch_size, X_dim)

            X, target = Variable(X), Variable(target)
            if cuda:
                X, target = X.cuda(), target.cuda()

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Reconstruction phase
            #######################
            if not labeled:
                z_sample = torch.cat(Q(X), 1)
                z_sample_noisy = z_sample + Variable(torch.randn(z_sample.size()) / 10.).cuda()
                X_sample = P(z_sample)

                recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
                recon_loss = recon_loss
                recon_loss.backward()
                P_decoder.step()
                Q_encoder.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()
                recon_loss = recon_loss
                #######################
                # Regularization phase
                #######################
                # Discriminator
                Q.eval()
                z_real_cat = sample_categorical(train_batch_size, n_classes=n_classes)
                z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))
                if cuda:
                    z_real_cat = z_real_cat.cuda()
                    z_real_gauss = z_real_gauss.cuda()

                z_fake_cat, z_fake_gauss = Q(X)

                D_real_cat = D_cat(z_real_cat)
                D_real_gauss = D_gauss(z_real_gauss)
                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) + torch.log(1 - D_fake_cat + TINY))
                D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

                D_loss = D_loss_cat + D_loss_gauss
                D_loss = D_loss

                D_loss.backward()
                D_cat_solver.step()
                D_gauss_solver.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()

                # Generator
                Q.train()
                z_fake_cat, z_fake_gauss = Q(X)

                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                G_loss = - torch.mean(torch.log(D_fake_cat + TINY)) - torch.mean(torch.log(D_fake_gauss + TINY))
                G_loss = G_loss
                G_loss.backward()
                Q_generator.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()

            #######################
            # Semi-supervised phase
            #######################
            if labeled:
                pred, _ = Q(X)
                class_loss = F.cross_entropy(pred, target)
                class_loss.backward()
                Q_semi_supervised.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()

    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss


def main():
    # Create and initialize Networks
    train_acc = []
    val_acc = []

    torch.manual_seed(10)
    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_cat = D_net_cat().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()
        D_cat = D_net_cat()

    # Set learning rates
    gen_lr = 0.001
    semi_lr = 0.001
    reg_lr = 0.0001

    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    Q_semi_supervised = optim.Adam(Q.parameters(), lr=semi_lr)

    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)
    D_cat_solver = optim.Adam(D_cat.parameters(), lr=reg_lr)

    for epoch in range(epochs):
        if epoch == 75:
            P_decoder = optim.Adam(P.parameters(), lr=gen_lr / 10.)
            Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr / 10.)

            Q_semi_supervised = optim.Adam(Q.parameters(), lr=semi_lr / 10.)

            Q_generator = optim.Adam(Q.parameters(), lr=reg_lr / 10.)
            D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr / 10.)
            D_cat_solver = optim.Adam(D_cat.parameters(), lr=reg_lr / 10.)

        D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss = train(P, Q, D_cat,
                                                                         D_gauss, P_decoder,
                                                                         Q_encoder, Q_semi_supervised,
                                                                         Q_generator,
                                                                         D_cat_solver, D_gauss_solver,
                                                                         train_labeled_loader,
                                                                         train_unlabeled_loader)
        if epoch % 10 == 0:
            train_acc.append(classification_accuracy(Q, train_labeled_loader))
            val_acc.append(classification_accuracy(Q, valid_loader))
            report_loss(epoch, D_loss_cat, D_loss_gauss, G_loss, recon_loss)
            print('Classification Loss: {:.3}'.format(class_loss.data[0]))
            print('Validation accuracy: {} %'.format(val_acc[-1]))


if __name__ == '__main__':
    main()
