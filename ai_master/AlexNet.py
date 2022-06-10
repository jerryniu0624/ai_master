import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一层是 5x5 的卷积，输入的channels 是 3，输出的channels是 64,步长 1,没有 padding
        # Conv2d 的第一个参数为输入通道，第二个参数为输出通道，第三个参数为卷积核大小
        # ReLU 的参数为inplace，True表示直接对输入进行修改，False表示创建新创建一个对象进行修改
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU()
        )

        # 第二层为 3x3 的池化，步长为2，没有padding
        self.max_pool1 = nn.MaxPool2d(3, 2)

        # 第三层是 5x5 的卷积， 输入的channels 是64，输出的channels 是64，没有padding
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True)
        )

        # 第四层是 3x3 的池化， 步长是 2，没有padding
        self.max_pool2 = nn.MaxPool2d(3, 2)

        # 第五层是全连接层，输入是 1204 ，输出是384
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 384),
            nn.ReLU(True)
        )

        # 第六层是全连接层，输入是 384， 输出是192
        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(True)
        )

        # 第七层是全连接层，输入是192， 输出是 10
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        # 将图片矩阵拉平
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        
