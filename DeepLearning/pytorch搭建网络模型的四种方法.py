
"""
  From: https://www.cnblogs.com/denny402/p/7593301.html
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict



# Method 1 -----------------------------------------

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

print("Method 1:")
model1 = Net1()
print(model1)


# Method 2 ------------------------------------------
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 2:")
model2 = Net2()
print(model2)



# Method 3 -------------------------------
class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv=torch.nn.Sequential()
        self.conv.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv.add_module("relu1",torch.nn.ReLU())
        self.conv.add_module("pool1",torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential()
        self.dense.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense.add_module("relu2",torch.nn.ReLU())
        self.dense.add_module("dense2",torch.nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 3:")
model3 = Net3()
print(model3)









# Method 4 ------------------------------------------
class Net4(torch.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", torch.nn.ReLU()),
                    ("pool", torch.nn.MaxPool2d(2))
                ]
            ))

        self.dense = torch.nn.Sequential(
            OrderedDict([
                ("dense1", torch.nn.Linear(32 * 3 * 3, 128)),
                ("relu2", torch.nn.ReLU()),
                ("dense2", torch.nn.Linear(128, 10))
            ])
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

print("Method 4:")
model4 = Net4()
print(model4)



