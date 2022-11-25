import numpy as np
import torch
import torchvision
import torchvision.transforms as tr
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt

transforms = tr.Compose([tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)

testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

classes = ('plane', ' car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def run():
    torch.multiporocessing.freeze_support()
    print('loop')

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr =0.001, momentum=0.9)

if __name__ == '__main__':
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            ## get the inputsl data is  a list of [input, labels]
            inputs, labels = data

            ## zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistic
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

print('finished Training')