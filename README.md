# AAE_pytorch

Adversarial Autoencoders in Pytorch


[intro]

## Background
##### Denoising Autoencoders (dAE)
The simplest version of an autoencoder is one in which we train a network to reconstruct its input. In other words, we would like the network to somehow learn the identity function $$f(x) = x$$. For this problem not to be trivial, we impose the condition of the network to go through an intermediate layer whose dimensionality is much lower than the dimensionality of the input. With this bottleneck condition, the network has to compress the input information in a way that it understands and can reconstruct afterwards. The network is therefore divided in two pieces, the *encoder* receives the input and creates a *latent* or *hidden* representation of it, and the *decoder* takes this intermediate representation and tries to reconstruct the input. The loss of an autoencoder is called \emph{recosntruction loss}, and can be defined simply as the squared error between the input and generated samples:

$L_R (x, x') = ||x - x'||^2$

\footnote{Another widely used reconstruction loss for the case when the input is normalized to be in the range [0,1]^N is the cross-entropy defined as 

Where $$x$$ is the input and $$x'$$ the output. 
##### Variational Autoencoders (VAE)
Variational autoencoders (VAE) impose a second constraint on how to construct the hidden representation. Now the latent code has a prior distribution defined by design. This also works as a regularization on the amount of information that can be stored in the latent. The benefit of this relies on the fact that now can use the system as a generative model. To create a new sample that comes from the data distribution $$p(x)$$, we just have to sample from $$p(z)$$ and run this sample through the *decoder* to *reconstruct* a new image. If this condition is not imposed, then the latent code can be distributed among the latent space freely and therefore is not possible to sample a new latent code to produce an image in a straightforward manner. 

In order to enforce this property a second term is added to the loss function in the form of a Kullback-Liebler (KL) divergence between the distribution created by the encoder and the prior distribution. Since VAE is based in a probabilistic interpretation, the reconstruction loss used is the cross-entropy loss. Putting this together we have, 

$$L(x, x') = L_R(x, x') + KL(q(z|x)||p(z)) = - \sum_{k=1}^N x_k \log(x'_k) + (1-x_k) \log(1-x'_k) + KL(q(z|x)|| p(z))$$

Where $$q(z|x)$$ is the encoder of our network and $$p(z)$$ is the prior distribution imposed on the latent code. 
(vae reference https://jaan.io/what-is-variational-autoencoder-vae-tutorial)
## Adversarial Autoencoders
One of the main drawbacks of variational autoencoders is that, the KL divergence term does not have a closed form solution except for a handful of distributions. Furthermore it is not straightforward to use discrete distributions on for $$z$$, one alternative for this is \ref{discrete variational autoencoders https://arxiv.org/abs/1609.02200}. 

Adversarial autoencodesr (AAE) avoid using the KL divergence altogether by using adversarial learning. Instead, a new network is trained to discriminatively predict whether a sample comes from the hidden code of the autoencoder or from the distribution $$p(z)$$ determined by the user. The loss of the encoder is now composed by the reconstruction loss plus the loss given by the discriminator network.

The image shows schematically how AAEs work. The top row is equivalent to an VAE. First a sample $$z$$ is drawn according to the generator network $$q(z|x)$$, that sample is then sent to the decoder which generates $$x'$$ from $$z$$. The reconstruction loss is computed between $$x$$ and $$x'$$ and the gradient is backpropagated through $$p$$ and $$q$$ accordingly and the weights of these to updated. 

On the adversarial regularization part the discriminator recieves $$z$$ distributed as $$q(z|x)$$ and $$z'$$ sampled from the true prior $$p(z)$$ and assigns a probability to each of coming from $$p(z)$$. The loss incurred is backpropagated through the discriminator first to update its weights. Then the process is repeated and the generator (encoder) updates its parameters.

We can now use the loss incurred by the the generator of the adversarial network (which is the encoder of the autoencoder) instead of a KL divergence for it to learn how to produce samples according to the distribution p(z). This modification allows us to use a broader set of distributions as priors for the latent code. 

Before getting into the training procedure used for this model, we look at some lines of how to implement what we have up to now in Pytorch. For the encoder, decoder and discriminator networks we will use simple feed forward neural networks with three 1000 hidden state layers with relu nonlinear functions and dropout.

``` Python
#Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.droppout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.droppout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))
```

Some things to note from this definitions. Frist, since the output of the encoder has to follow a Gaussian distribution, we do not use any nonlinearities at its last output. Also, the output of the decoder has a sigmoid nonlinearity, this is because we are using the inputs normalized in a way in which their values are within $$0$$ and $$1$$. The output of the discriminator network is just one number between $$0$$ and $$1$$ representing the probability of the input coming from the true prior distribution. 

Once the networks classes are defined, we create an instance of each one and define the optipmizers to be used. In order to have independence in the optimization procedure for the encoder (which is as well the generator of the adversarial network) we define two optimizers for this part of the network as follows:

``` Python
    torch.manual_seed(10)
    Q, P = Q_net() = Q_net(), P_net(0)     # Encoder/Decoder
    D_gauss = D_net_gauss()     # Discriminator adversarial
    if torch.cuda.is_available():
        Q = Q.cuda()
        P = P.cuda()
        D_cat = D_gauss.cuda()
        D_gauss = D_net_gauss().cuda()
    
    # Set learning rates
    gen_lr, reg_lr = 0.0006, 0.0008
    
    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)
    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)
```

The training procedure for this architecture for each minibatch is performed as follows:
1. Do a forward path through the encoder/decoder part, compute and update the parameteres of the encoder Q and decoder P networks. 
``` Python
    z_sample = Q(X)
    X_sample = P(z_sample)
    recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
    recon_loss.backward()
    P_decoder.step()
    Q_encoder.step()
```
2. Create a latent representation z = Q(x) and a sample z’ from the prior p(z), run each one through the discriminator and compute the score assigned to each (D(z) and D(z’)). 
``` Python
    Q.eval()    
    z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))
    if torch.cuda.is_available():
        z_real_gauss = z_real_gauss.cuda()
    z_fake_gauss = Q(X)

    D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)
    D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
    D_loss.backward()
    D_gauss_solver.step()
```
3. Compute the loss in the discriminator as $$L_D(z, z’) = - \frac{1}{m} \sum_k \log(D(z’)) +  \log(1-D(z))$$ and backpropagate it through the discriminator network to update its weights. The scalar $m$ corresponds to the minibatch size. In code, 
``` Python
        Q.eval()    # Not use dropout in Q in this step
        z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))

        if cuda:
            z_real_gauss = z_real_gauss.cuda()

        z_fake_gauss = Q(X)

        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_gauss = D_gauss(z_fake_gauss)
        D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

        D_loss.backward()
        D_gauss_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()
```
4. Compute the loss of the generator network as $L_G(z) = -\frac{1}{m} \sum_k \log D(z)$ and update Q network accordingly.
``` Python
        # Generator
        Q.train()   # Back to use dropout
        z_fake_gauss = Q(X)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
        G_loss.backward()
        Q_generator.step()
```


