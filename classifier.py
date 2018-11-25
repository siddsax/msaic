import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class classifier1(nn.Module):
    def __init__(self):
        super(classifier1, self).__init__()
        self.convA1 = nn.Conv2d(1, 4, (3, 10))
        self.maxPA1 = nn.MaxPool2d((2,3), (2, 3))
        self.convA2 = nn.Conv2d(4, 2, (2, 4))
        self.maxPA2 = nn.MaxPool2d((2,2), (2, 2))
        self.denseA1 = nn.Linear(20, 4)

        self.convB1 = nn.Conv2d(1, 4, (5, 10))
        self.maxPB1 = nn.MaxPool2d((5, 5), (5, 5))
        self.convB2 = nn.Conv2d(4, 2, (3, 3))
        self.maxPB2 = nn.MaxPool2d((2,2), (2, 2))
        self.denseB1 = nn.Linear(18, 4)

        self.denseAB = nn.Linear(4, 1)

    def forward(self, queryfeatures, passagefeatures):
        x1 = F.tanh(self.convA1(queryfeatures))
        x1 = F.tanh(self.maxPA1(x1))
        x1 = F.tanh(self.convA2(x1))
        x1 = F.tanh(self.maxPA2(x1))
        x1 = x1.view(x1.shape[0], -1)
        x1 = F.tanh(self.denseA1(x1))

        x2 = F.tanh(self.convB1(passagefeatures))
        x2 = F.tanh(self.maxPB1(x2))
        x2 = F.tanh(self.convB2(x2))
        x2 = F.tanh(self.maxPB2(x2))
        x2 = x2.view(x2.shape[0], -1)
        x2 = F.tanh(self.denseB1(x2))

        # x = torch.cat((x1, x2), dim=-1)
        x = x1*x2
        x = self.denseAB(x)

        return F.sigmoid(x)

