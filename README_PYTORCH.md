# PyTorch 多文件项目示例

这是一个完整的 PyTorch 多文件项目示例，包含多种常用模型和训练工具，适合在 Google Colab 上进行多文件测试。

## 📁 项目结构

```
UCAS/
├── models/                  # 模型目录
│   ├── __init__.py         # 模型包初始化
│   ├── simple_nn.py        # 简单神经网络（MLP）
│   ├── cnn.py              # 卷积神经网络
│   ├── resnet.py           # ResNet 残差网络
│   └── lstm.py             # LSTM 循环神经网络
├── utils/                   # 工具目录
│   ├── __init__.py         # 工具包初始化
│   ├── data_loader.py      # 数据加载工具
│   └── train_utils.py      # 训练工具函数
├── train.py                 # 训练主脚本
├── colab_demo.ipynb        # Colab 演示 Notebook
├── requirements.txt        # 依赖包列表
└── README_PYTORCH.md       # 本文件
```

## 🚀 快速开始

### 1. 本地使用

```bash
# 安装依赖
pip install -r requirements.txt

# 运行训练脚本
python train.py --model simple_nn  # 训练简单神经网络
python train.py --model cnn        # 训练 CNN
python train.py --model resnet     # 训练 ResNet
python train.py --model lstm       # 训练 LSTM
python train.py --model all        # 训练所有模型
```

### 2. 在 Google Colab 中使用

#### 方法 A: 从 GitHub 克隆

```python
# 在 Colab 单元格中运行
!git clone https://github.com/your-username/UCAS.git
%cd UCAS
!pip install -r requirements.txt
```

#### 方法 B: 手动上传文件

1. 将整个项目文件夹压缩为 ZIP
2. 在 Colab 中运行以下代码：

```python
from google.colab import files
uploaded = files.upload()  # 上传 ZIP 文件

!unzip UCAS.zip
%cd UCAS
!pip install -r requirements.txt
```

#### 方法 C: 使用 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/UCAS
!pip install -r requirements.txt
```

### 3. 使用 Jupyter Notebook

打开 `colab_demo.ipynb` 文件，按照步骤执行即可。

## 📚 模型介绍

### 1. SimpleNN (`models/simple_nn.py`)

简单的多层感知机（MLP），适用于：
- MNIST 手写数字识别
- 简单的分类任务
- 学习基础的神经网络概念

**特点**：
- 可配置的隐藏层数量和大小
- Batch Normalization
- Dropout 正则化

**示例代码**：
```python
from models import SimpleNN

model = SimpleNN(
    input_size=784,
    hidden_sizes=[512, 256, 128],
    num_classes=10,
    dropout=0.5
)
```

### 2. SimpleCNN & AdvancedCNN (`models/cnn.py`)

卷积神经网络，适用于：
- 图像分类（CIFAR-10, CIFAR-100）
- 特征提取
- 计算机视觉任务

**特点**：
- SimpleCNN：基础 CNN 架构
- AdvancedCNN：深层网络 + Batch Normalization

**示例代码**：
```python
from models import SimpleCNN, AdvancedCNN

# 简单 CNN
model = SimpleCNN(num_classes=10, in_channels=3)

# 高级 CNN
model = AdvancedCNN(num_classes=10, in_channels=3)
```

### 3. ResNet (`models/resnet.py`)

残差网络，适用于：
- 深度图像分类
- 迁移学习
- 特征提取

**特点**：
- ResNet-18 和 ResNet-34 两种配置
- 残差连接解决梯度消失问题
- 适应性强

**示例代码**：
```python
from models import ResNet18, ResNet34

# ResNet-18
model = ResNet18(num_classes=10, in_channels=3)

# ResNet-34
model = ResNet34(num_classes=10, in_channels=3)
```

### 4. SimpleLSTM & BidirectionalLSTM (`models/lstm.py`)

循环神经网络，适用于：
- 文本分类
- 序列预测
- 时间序列分析

**特点**：
- SimpleLSTM：单向 LSTM
- BidirectionalLSTM：双向 LSTM + 注意力机制

**示例代码**：
```python
from models import SimpleLSTM, BidirectionalLSTM

# 简单 LSTM
model = SimpleLSTM(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    num_classes=5
)

# 双向 LSTM
model = BidirectionalLSTM(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    num_classes=5
)
```

## 🛠️ 工具函数

### 数据加载 (`utils/data_loader.py`)

提供常用数据集的加载函数：

```python
from utils import get_mnist_loaders, get_cifar10_loaders, create_synthetic_dataset

# MNIST
train_loader, test_loader = get_mnist_loaders(batch_size=64)

# CIFAR-10
train_loader, test_loader = get_cifar10_loaders(batch_size=128, augment=True)

# 合成数据（用于快速测试）
train_loader, test_loader = create_synthetic_dataset(num_samples=1000)
```

### 训练工具 (`utils/train_utils.py`)

提供完整的训练、验证、测试流程：

```python
from utils import train_epoch, validate, test, save_checkpoint, load_checkpoint
from utils import EarlyStopping, MetricTracker

# 训练一个 epoch
train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

# 验证
val_loss, val_acc = validate(model, val_loader, criterion, device)

# 测试
test_acc, predictions, labels = test(model, test_loader, device)

# 保存模型
save_checkpoint(model, optimizer, epoch, loss, acc, 'checkpoint.pth')

# 加载模型
epoch, loss, acc = load_checkpoint(model, optimizer, 'checkpoint.pth', device)

# 早停机制
early_stopping = EarlyStopping(patience=5)
early_stopping(val_loss)

# 指标追踪
tracker = MetricTracker()
tracker.update(train_loss, train_acc, val_loss, val_acc)
tracker.print_summary()
```

## 📊 完整训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet18
from utils import get_cifar10_loaders, train_epoch, validate, MetricTracker

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# 创建模型
model = ResNet18(num_classes=10).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练
num_epochs = 100
tracker = MetricTracker()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    
    tracker.update(train_loss, train_acc, val_loss, val_acc)
    scheduler.step()
    
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

# 打印总结
tracker.print_summary()
```

## 💡 Colab 使用技巧

### 1. 检查 GPU

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2. 挂载 Google Drive（保存模型）

```python
from google.colab import drive
drive.mount('/content/drive')

# 保存到 Drive
save_checkpoint(model, optimizer, epoch, loss, acc, 
                '/content/drive/MyDrive/models/checkpoint.pth')
```

### 3. 监控训练进度

```python
# 使用 TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs

# 或使用 tqdm 进度条（已集成在 train_epoch 中）
```

### 4. 下载训练好的模型

```python
from google.colab import files
files.download('checkpoint.pth')
```

## 📝 测试各个模块

```python
# 测试模型导入
from models import SimpleNN, SimpleCNN, ResNet18, SimpleLSTM
print("✅ 模型导入成功")

# 测试数据加载
from utils import create_synthetic_dataset
train_loader, test_loader = create_synthetic_dataset()
print("✅ 数据加载成功")

# 测试训练工具
from utils import MetricTracker
tracker = MetricTracker()
print("✅ 工具函数导入成功")

# 测试模型前向传播
model = SimpleNN()
x = torch.randn(32, 1, 28, 28)
output = model(x)
print(f"✅ 模型前向传播成功，输出形状: {output.shape}")
```

## 🎯 学习路径建议

1. **初学者**：
   - 从 `SimpleNN` 和 `simple_nn.py` 开始
   - 使用合成数据快速测试
   - 理解训练循环的基本流程

2. **进阶者**：
   - 学习 `SimpleCNN` 和卷积操作
   - 尝试 MNIST 和 CIFAR-10 数据集
   - 实验不同的优化器和学习率

3. **高级用户**：
   - 研究 `ResNet` 的残差连接
   - 探索 LSTM 在序列数据上的应用
   - 实现自定义模型和数据增强

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可证

MIT License

## 📮 联系方式

如有问题，请通过 Issue 或 Email 联系。

---

**祝你学习愉快！🎉**

