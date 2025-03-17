import torch
import torch.nn as nn
import torch.nn.functional as F

# F.mse_loss(input, target) = 1/n * (input - target)^2 ，均方误差
# 继承nn.Module，重构forward方法，可以直接加入model的层

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        # 求出欧氏距离
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    # image = (batch, channel, height, width)
    a, b, c, d = input.size()  
    # 按通道展开为(3, height * width)的二维矩阵计算Gram矩阵
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product
    # 归一化处理
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input