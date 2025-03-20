import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,ResNet18_Weights

# 浅层UNet分支（4层结构）
class ShallowUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 解码器
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 跳跃连接融合
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.constant_(m.weight, 0)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        e1 = self.enc1(x)       # [B, 64, H, W]
        e2 = self.enc2(e1)      # [B, 128, H/2, W/2]
        d1 = self.up1(e2)       # [B, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)  # 跳跃连接
        return self.dec1(d1)    # [B, 64, H, W]

# SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 双分支ResNet-18 + UNet模型
class DualPathResNet18_UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 主分支：ResNet-18（截断到layer2）
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_encoder = nn.Sequential(
            resnet.conv1,       # [B,64,128,128] (输入256x256时)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,     # [B,64,64,64]
            resnet.layer1,      # [B,64,64,64] (2个残差块)
            resnet.layer2       # [B,128,32,32] (2个残差块)
        )
        
        # 辅助分支：浅层UNet
        self.unet_branch = ShallowUNet(in_channels=3)
        
        # 特征融合模块（含SE注意力）
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 1),  # 通道对齐（128+64→128）
            SEBlock(128),                 # 通道注意力
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 主分支后续层（ResNet-18的layer3和layer4）
        self.resnet_tail = nn.Sequential(
            resnet.layer3,  # [B,256,16,16] (2个残差块)
            resnet.layer4   # [B,512,8,8] (2个残差块)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 主分支前向
        resnet_feat = self.resnet_encoder(x)  # [B,128,32,32]
        
        # 辅助分支前向
        unet_feat = self.unet_branch(x)       # [B,64,256,256]
        unet_feat = F.interpolate(unet_feat, size=32, mode='bilinear')  # 下采样到32x32
        
        # 特征融合（带SE注意力）
        fused = torch.cat([resnet_feat, unet_feat], dim=1)  # [B,192,32,32]
        fused = self.fusion(fused)            # [B,256,32,32]
        
        # 主分支继续前向
        out = self.resnet_tail(fused)         # 经过layer3和layer4
        return self.classifier(out)

if __name__ == "__main__":
    # 测试双分支模型
    model = DualPathResNet18_UNet(num_classes=10)
    x = torch.randn(1, 3, 256, 256)  # 输入图像
    out = model(x)
    print(out.shape)  # 输出分类结果
