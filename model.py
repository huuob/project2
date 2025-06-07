import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_Deep(nn.Module):
    def __init__(self, activation="relu"):
        super(SimpleCNN_Deep, self).__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        else:
            return x

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
