from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


class subMNIST(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, k=3000):
        super(subMNIST, self).__init__(root, train, transform,
                                       target_transform, download)
        self.k = k

    def __len__(self):
        if self.train:
            return self.k
        else:
            return 10000

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
trainset = subMNIST(root='../data', train=True,
                    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


print(len(trainset))
print(len(trainloader))
