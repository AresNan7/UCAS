"""
数据加载工具
提供常用数据集的加载函数
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


def get_mnist_loaders(batch_size=64, num_workers=2, data_dir='./data'):
    """
    获取 MNIST 数据加载器
    
    参数:
        batch_size: 批次大小
        num_workers: 数据加载线程数
        data_dir: 数据存储目录
    
    返回:
        train_loader, test_loader
    """
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=64, num_workers=2, data_dir='./data', augment=True):
    """
    获取 CIFAR-10 数据加载器
    
    参数:
        batch_size: 批次大小
        num_workers: 数据加载线程数
        data_dir: 数据存储目录
        augment: 是否使用数据增强
    
    返回:
        train_loader, test_loader
    """
    # 训练数据转换
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # 测试数据转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 训练集
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 测试集
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


class CustomDataset(Dataset):
    """
    自定义数据集类模板
    """
    def __init__(self, data, labels, transform=None):
        """
        参数:
            data: 输入数据
            labels: 标签
            transform: 数据转换
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


def create_synthetic_dataset(num_samples=1000, input_dim=10, num_classes=3):
    """
    创建合成数据集用于测试
    
    参数:
        num_samples: 样本数量
        input_dim: 输入维度
        num_classes: 类别数
    
    返回:
        train_loader, test_loader
    """
    # 生成随机数据
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples).astype(np.int64)
    
    # 划分训练集和测试集
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 转换为 Tensor
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    print("测试数据加载器...")
    
    # 测试合成数据集
    print("\n创建合成数据集...")
    train_loader, test_loader = create_synthetic_dataset()
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 获取一个批次
    data, labels = next(iter(train_loader))
    print(f"数据形状: {data.shape}")
    print(f"标签形状: {labels.shape}")

