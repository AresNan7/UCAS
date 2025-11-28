"""
卷积神经网络模型
包含简单 CNN 和高级 CNN 两种架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络
    适用于 MNIST 等简单图像分类任务
    """
    def __init__(self, num_classes=10, in_channels=1):
        """
        参数:
            num_classes: 输出类别数
            in_channels: 输入通道数（灰度图为1，RGB为3）
        """
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层（需要根据输入大小调整）
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 卷积层 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # 卷积层 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # 卷积层 3
        x = self.pool(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class AdvancedCNN(nn.Module):
    """
    高级卷积神经网络
    包含 Batch Normalization 和更深的网络结构
    """
    def __init__(self, num_classes=10, in_channels=3):
        """
        参数:
            num_classes: 输出类别数
            in_channels: 输入通道数
        """
        super(AdvancedCNN, self).__init__()
        
        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 第二个卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 第三个卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # 测试 SimpleCNN
    print("=" * 50)
    print("测试 SimpleCNN")
    print("=" * 50)
    model = SimpleCNN(num_classes=10, in_channels=1)
    test_input = torch.randn(4, 1, 28, 28)
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试 AdvancedCNN
    print("\n" + "=" * 50)
    print("测试 AdvancedCNN")
    print("=" * 50)
    model = AdvancedCNN(num_classes=10, in_channels=3)
    test_input = torch.randn(4, 3, 32, 32)
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")

