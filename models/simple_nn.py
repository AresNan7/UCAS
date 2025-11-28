"""
简单的全连接神经网络
适用于 MNIST、CIFAR-10 等基础数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    简单的多层感知机（MLP）
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout=0.5):
        """
        参数:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表
            num_classes: 输出类别数
            dropout: Dropout 比率
        """
        super(SimpleNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        """
        # 展平输入
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x
    
    def get_num_params(self):
        """
        获取模型参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    model = SimpleNN()
    print(f"模型结构:\n{model}")
    print(f"\n可训练参数数量: {model.get_num_params():,}")
    
    # 测试前向传播
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")

