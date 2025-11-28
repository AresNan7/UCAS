# 快速入门指南

## 🚀 在 Google Colab 中使用

### 方法 1: 上传文件

1. 将整个 UCAS 文件夹压缩为 ZIP 文件
2. 在 Colab 中创建新 Notebook
3. 运行以下代码：

```python
from google.colab import files
uploaded = files.upload()  # 上传 ZIP 文件

!unzip UCAS.zip
%cd UCAS
!pip install -q torch torchvision torchaudio tqdm numpy matplotlib
```

### 方法 2: 从 GitHub 克隆（推荐）

```python
!git clone https://github.com/your-username/UCAS.git
%cd UCAS
!pip install -q torch torchvision torchaudio tqdm numpy matplotlib
```

### 方法 3: 使用 Google Drive

1. 将 UCAS 文件夹上传到 Google Drive
2. 在 Colab 中运行：

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/UCAS
!pip install -q torch torchvision torchaudio tqdm numpy matplotlib
```

## 📝 基础测试

### 测试 1: 验证模块安装

```python
# 运行测试脚本
!python test_modules.py
```

### 测试 2: 导入所有模块

```python
# 导入模型
from models import SimpleNN, SimpleCNN, ResNet18, SimpleLSTM

# 导入工具
from utils import train_epoch, validate, create_synthetic_dataset

print("✅ 所有模块导入成功！")
```

### 测试 3: 创建并测试一个模型

```python
import torch
from models import SimpleNN

# 创建模型
model = SimpleNN(input_size=784, hidden_sizes=[256, 128], num_classes=10)
print(f"模型参数量: {model.get_num_params():,}")

# 测试前向传播
x = torch.randn(32, 1, 28, 28)
output = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
```

## 🎯 快速训练示例

### 示例 1: 使用合成数据快速训练

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models import SimpleNN
from utils import create_synthetic_dataset, train_epoch, validate

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建合成数据
train_loader, test_loader = create_synthetic_dataset(
    num_samples=1000, 
    input_dim=20, 
    num_classes=5
)

# 创建模型
model = SimpleNN(input_size=20, hidden_sizes=[64, 32], num_classes=5).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练 10 个 epoch
for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, verbose=False)
    val_loss, val_acc = validate(model, test_loader, criterion, device, verbose=False)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
```

### 示例 2: 训练 CNN（需要下载数据）

```python
import torch.nn as nn
import torch.optim as optim
from models import SimpleCNN
from utils import get_cifar10_loaders, train_epoch, validate

# 加载 CIFAR-10 数据集
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# 创建模型
model = SimpleCNN(num_classes=10, in_channels=3).to(device)

# 训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}: Val Acc={val_acc:.2f}%")
```

## 🎓 使用训练脚本

### 运行单个模型

```bash
# 训练简单神经网络
python train.py --model simple_nn

# 训练 CNN
python train.py --model cnn

# 训练 ResNet
python train.py --model resnet

# 训练 LSTM
python train.py --model lstm
```

### 运行所有模型

```bash
python train.py --model all
```

## 📊 可视化训练结果

```python
import matplotlib.pyplot as plt
from utils import MetricTracker

# 假设你已经训练了模型并使用 MetricTracker
tracker = MetricTracker()
# ... 训练过程 ...

# 绘制曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(tracker.train_losses, label='Train Loss')
plt.plot(tracker.val_losses, label='Val Loss')
plt.legend()
plt.title('Loss Curves')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tracker.train_accs, label='Train Acc')
plt.plot(tracker.val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy Curves')
plt.grid(True)

plt.show()
```

## 💾 保存和加载模型

```python
from utils import save_checkpoint, load_checkpoint

# 保存模型
save_checkpoint(
    model, 
    optimizer, 
    epoch=10, 
    loss=0.1, 
    acc=95.0, 
    filepath='my_model.pth'
)

# 加载模型
epoch, loss, acc = load_checkpoint(
    model, 
    optimizer, 
    filepath='my_model.pth', 
    device=device
)
```

## 🔧 常见问题

### 问题 1: 导入模块失败

**解决方案**: 确保你在正确的目录下

```python
import os
print(os.getcwd())  # 应该显示 UCAS 目录

# 如果不在 UCAS 目录，运行：
%cd /path/to/UCAS
```

### 问题 2: CUDA 内存不足

**解决方案**: 减小 batch size

```python
# 使用更小的 batch size
train_loader, test_loader = get_cifar10_loaders(batch_size=32)  # 而不是 128
```

### 问题 3: 训练速度慢

**解决方案**: 
1. 确保使用 GPU: Runtime → Change runtime type → GPU
2. 减少数据量进行快速测试
3. 使用更小的模型

## 📚 学习资源

### 文件说明

- `models/simple_nn.py` - 多层感知机，适合初学者
- `models/cnn.py` - 卷积神经网络，图像分类
- `models/resnet.py` - 残差网络，深度学习
- `models/lstm.py` - 循环神经网络，序列数据
- `utils/data_loader.py` - 数据加载工具
- `utils/train_utils.py` - 训练辅助函数
- `train.py` - 完整训练脚本
- `colab_demo.ipynb` - Colab 演示 Notebook
- `test_modules.py` - 模块测试脚本

### 学习路径

1. **第一步**: 运行 `test_modules.py` 验证环境
2. **第二步**: 打开 `colab_demo.ipynb` 跟随教程
3. **第三步**: 阅读 `models/simple_nn.py` 理解基础模型
4. **第四步**: 尝试修改超参数，观察结果
5. **第五步**: 实现自己的模型

## 🎉 开始使用

现在你可以开始使用了！建议从 `colab_demo.ipynb` 开始，它包含了完整的示例代码。

查看详细文档：[README_PYTORCH.md](README_PYTORCH.md)

