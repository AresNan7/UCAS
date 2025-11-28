# UCAS
UCAS learning

## PyTorch 多文件项目

这个项目包含多个 PyTorch 模型和训练脚本，适合在 Google Colab 上进行多文件测试。

### 📂 项目结构

```
UCAS/
├── models/              # 深度学习模型
│   ├── simple_nn.py    # 简单全连接网络
│   ├── cnn.py          # 卷积神经网络
│   ├── resnet.py       # ResNet 残差网络
│   └── lstm.py         # LSTM 循环神经网络
├── utils/              # 工具函数
│   ├── data_loader.py  # 数据加载
│   └── train_utils.py  # 训练工具
├── train.py            # 训练脚本
├── colab_demo.ipynb    # Colab 演示
└── requirements.txt    # 依赖包
```

### 🚀 快速开始

1. **本地运行**：
```bash
pip install -r requirements.txt
python train.py --model simple_nn
```

2. **在 Colab 中使用**：
   - 打开 `colab_demo.ipynb`
   - 或参考 `README_PYTORCH.md` 获取详细说明

### 📚 包含的模型

- **SimpleNN**: 多层感知机（MLP）
- **SimpleCNN / AdvancedCNN**: 卷积神经网络
- **ResNet-18 / ResNet-34**: 残差网络
- **SimpleLSTM / BidirectionalLSTM**: 循环神经网络

详细文档请查看 [README_PYTORCH.md](README_PYTORCH.md)
