import torch.nn as nn

# 未使用下面的反射填充方法的版本
class geneNet(nn.Module):
    def __init__(self):
        super(geneNet, self).__init__()
        # 下卷积采样模块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # 残差块
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # 上卷积恢复模块
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        # 非线性激活函数输出
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
        

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        res = x
        out = self.block(x)
        out = out + res
        return out

# 使用反射填充，代替使用0填充的基本填充方式，使其对于边缘敏感的特征提取更加有效
class Convlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Convlayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
    
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out