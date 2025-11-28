"""
训练脚本
演示如何使用各种模型和工具进行训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models import SimpleNN, SimpleCNN, ResNet18, SimpleLSTM
from utils import get_mnist_loaders, get_cifar10_loaders, create_synthetic_dataset
from utils import train_epoch, validate, test, save_checkpoint, EarlyStopping, MetricTracker
import argparse
import os


def train_simple_nn():
    """训练简单神经网络（MNIST）"""
    print("\n" + "=" * 60)
    print("训练简单神经网络 (SimpleNN on MNIST)")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载 MNIST 数据集...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir='./data')
    
    # 创建模型
    model = SimpleNN(input_size=784, hidden_sizes=[512, 256], num_classes=10).to(device)
    print(f"\n模型参数数量: {model.get_num_params():,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    num_epochs = 5
    tracker = MetricTracker()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        tracker.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
    
    # 测试
    print("\n最终测试...")
    test_acc, _, _ = test(model, test_loader, device)
    print(f"测试准确率: {test_acc:.2f}%")
    
    # 保存模型
    save_checkpoint(model, optimizer, num_epochs, val_loss, val_acc, 
                   './checkpoints/simple_nn_mnist.pth')
    
    tracker.print_summary()


def train_cnn():
    """训练卷积神经网络（CIFAR-10）"""
    print("\n" + "=" * 60)
    print("训练卷积神经网络 (SimpleCNN on CIFAR-10)")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载 CIFAR-10 数据集...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, data_dir='./data')
    
    # 创建模型
    model = SimpleCNN(num_classes=10, in_channels=3).to(device)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练
    num_epochs = 10
    tracker = MetricTracker()
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        tracker.update(train_loss, train_acc, val_loss, val_acc)
        scheduler.step()
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("触发早停机制!")
            break
    
    # 测试
    print("\n最终测试...")
    test_acc, _, _ = test(model, test_loader, device)
    print(f"测试准确率: {test_acc:.2f}%")
    
    # 保存模型
    save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc,
                   './checkpoints/cnn_cifar10.pth')
    
    tracker.print_summary()


def train_resnet():
    """训练 ResNet（CIFAR-10）"""
    print("\n" + "=" * 60)
    print("训练 ResNet-18 (CIFAR-10)")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载 CIFAR-10 数据集...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, data_dir='./data', augment=True)
    
    # 创建模型
    model = ResNet18(num_classes=10, in_channels=3).to(device)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 训练
    num_epochs = 3  # 演示用，实际训练需要更多 epoch
    tracker = MetricTracker()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        tracker.update(train_loss, train_acc, val_loss, val_acc)
        scheduler.step()
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 测试
    print("\n最终测试...")
    test_acc, _, _ = test(model, test_loader, device)
    print(f"测试准确率: {test_acc:.2f}%")
    
    # 保存模型
    save_checkpoint(model, optimizer, num_epochs, val_loss, val_acc,
                   './checkpoints/resnet18_cifar10.pth')
    
    tracker.print_summary()


def train_lstm():
    """训练 LSTM（合成数据）"""
    print("\n" + "=" * 60)
    print("训练 LSTM (合成序列数据)")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建合成序列数据
    print("\n创建合成序列数据...")
    vocab_size = 1000
    seq_length = 50
    num_samples = 5000
    num_classes = 5
    
    # 生成随机序列数据
    train_data = torch.randint(0, vocab_size, (int(num_samples * 0.8), seq_length))
    train_labels = torch.randint(0, num_classes, (int(num_samples * 0.8),))
    test_data = torch.randint(0, vocab_size, (int(num_samples * 0.2), seq_length))
    test_labels = torch.randint(0, num_classes, (int(num_samples * 0.2),))
    
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建模型
    model = SimpleLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256,
                      num_layers=2, num_classes=num_classes).to(device)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    num_epochs = 5
    tracker = MetricTracker()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练循环
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * correct / total
        
        tracker.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
    
    # 保存模型
    save_checkpoint(model, optimizer, num_epochs, val_loss, val_acc,
                   './checkpoints/lstm_synthetic.pth')
    
    tracker.print_summary()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch 模型训练脚本')
    parser.add_argument('--model', type=str, default='all',
                       choices=['simple_nn', 'cnn', 'resnet', 'lstm', 'all'],
                       help='选择要训练的模型')
    
    args = parser.parse_args()
    
    # 创建检查点目录
    os.makedirs('./checkpoints', exist_ok=True)
    
    print("PyTorch 多模型训练演示")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    # 根据参数训练相应模型
    if args.model == 'simple_nn' or args.model == 'all':
        train_simple_nn()
    
    if args.model == 'cnn' or args.model == 'all':
        train_cnn()
    
    if args.model == 'resnet' or args.model == 'all':
        train_resnet()
    
    if args.model == 'lstm' or args.model == 'all':
        train_lstm()
    
    print("\n" + "=" * 60)
    print("所有训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

