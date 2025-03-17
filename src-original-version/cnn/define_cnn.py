import torch
from torch import nn

class MyStyleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MyStyleCNN, self).__init__()

        #定义多个包含卷积层、激活函数和池化层的卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二个卷积块：输出通道数设为64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第三个卷积块：输出通道数设为128
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 添加全连接层帮助模型训练
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = {}
        # 依次经过三个卷积块
        x = self.conv_block1(x)
        features['conv1'] = x

        x = self.conv_block2(x)
        features['conv2'] = x

        x = self.conv_block3(x)
        features['conv3'] = x
        # 对height和width维度取平均值，全局平均池化得到特征向量,每个通道对应一个值
        x = x.mean([2, 3]) # shape: [batch, 128]
        logits = self.classifier(x)
        return logits, features