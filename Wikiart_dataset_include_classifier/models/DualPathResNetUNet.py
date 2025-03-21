import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

# 浅层UNet分支（示例4层结构）
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
        # 编码
        e1 = self.enc1(x)      # [B, 64, H, W]
        e2 = self.enc2(e1)     # [B, 128, H/2, W/2]
        # 解码
        d1 = self.up1(e2)      # [B, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)  # 跳跃连接
        out = self.dec1(d1)    # [B, 64, H, W]
        return out

# 双分支融合模型
class DualPathResNetUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 主分支：ResNet-34（截断到layer2）
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet_encoder = nn.Sequential(
            resnet.conv1,       # [B, 64, 128, 128]（输入256x256时）
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,      # [B, 64, 64, 64]
            resnet.layer1,       # [B, 64, 64, 64]
            resnet.layer2        # [B, 128, 32, 32]
        )
        
        # 辅助分支：浅层UNet
        self.unet_branch = ShallowUNet(in_channels=3)
        
        # 特征融合模块（将UNet输出与ResNet的layer2输出融合）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 + 128, 128, 1),  # 调整通道数
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 主分支后续层（ResNet的layer3和layer4）
        self.resnet_tail = nn.Sequential(
            resnet.layer3,       # [B, 256, 16, 16]
            resnet.layer4        # [B, 512, 8, 8]
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 主分支前向（到layer2）
        resnet_feat = self.resnet_encoder(x)  # [B, 128, 32, 32]
        
        # 辅助分支前向（UNet）
        unet_feat = self.unet_branch(x)       # [B, 64, 256, 256]
        
        # 特征对齐与融合
        # Step 1: 对UNet输出下采样到32x32
        unet_feat = nn.functional.interpolate(unet_feat, size=(32, 32), mode='bilinear')
        # Step 2: 拼接特征（通道维度）
        fused_feat = torch.cat([resnet_feat, unet_feat], dim=1)  # [B, 128+64=192, 32, 32]
        fused_feat = self.fusion_conv(fused_feat)                # [B, 256, 32, 32]
        
        # 主分支继续前向
        out = self.resnet_tail(fused_feat)    # 经过layer3和layer4
        out = self.classifier(out)
        return out


if __name__=="__main__":
    # 使用示例
    model = DualPathResNetUNet(num_classes=11)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)  # torch.Size([1, 10])