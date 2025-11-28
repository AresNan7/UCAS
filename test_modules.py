"""
快速测试脚本
用于验证所有模块是否正常工作
"""

import torch
import sys


def test_models():
    """测试所有模型"""
    print("=" * 60)
    print("测试模型模块")
    print("=" * 60)
    
    try:
        from models import SimpleNN, SimpleCNN, AdvancedCNN, ResNet18, ResNet34, SimpleLSTM, BidirectionalLSTM
        print("✅ 所有模型导入成功")
        
        # 测试 SimpleNN
        model = SimpleNN(input_size=784, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10), "SimpleNN 输出形状错误"
        print(f"✅ SimpleNN 测试通过 (参数量: {model.get_num_params():,})")
        
        # 测试 SimpleCNN
        model = SimpleCNN(num_classes=10, in_channels=1)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10), "SimpleCNN 输出形状错误"
        print(f"✅ SimpleCNN 测试通过 (参数量: {sum(p.numel() for p in model.parameters()):,})")
        
        # 测试 AdvancedCNN
        model = AdvancedCNN(num_classes=10, in_channels=3)
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == (4, 10), "AdvancedCNN 输出形状错误"
        print(f"✅ AdvancedCNN 测试通过 (参数量: {sum(p.numel() for p in model.parameters()):,})")
        
        # 测试 ResNet18
        model = ResNet18(num_classes=10, in_channels=3)
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == (4, 10), "ResNet18 输出形状错误"
        print(f"✅ ResNet18 测试通过 (参数量: {sum(p.numel() for p in model.parameters()):,})")
        
        # 测试 SimpleLSTM
        vocab_size = 1000
        model = SimpleLSTM(vocab_size=vocab_size, num_classes=5)
        x = torch.randint(0, vocab_size, (4, 50))
        output, hidden = model(x)
        assert output.shape == (4, 5), "SimpleLSTM 输出形状错误"
        print(f"✅ SimpleLSTM 测试通过 (参数量: {sum(p.numel() for p in model.parameters()):,})")
        
        # 测试 BidirectionalLSTM
        model = BidirectionalLSTM(vocab_size=vocab_size, num_classes=5)
        x = torch.randint(0, vocab_size, (4, 50))
        output = model(x)
        assert output.shape == (4, 5), "BidirectionalLSTM 输出形状错误"
        print(f"✅ BidirectionalLSTM 测试通过 (参数量: {sum(p.numel() for p in model.parameters()):,})")
        
        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """测试工具模块"""
    print("\n" + "=" * 60)
    print("测试工具模块")
    print("=" * 60)
    
    try:
        from utils import (
            create_synthetic_dataset, 
            train_epoch, validate, test,
            save_checkpoint, load_checkpoint,
            EarlyStopping, MetricTracker
        )
        print("✅ 所有工具函数导入成功")
        
        # 测试数据加载
        train_loader, test_loader = create_synthetic_dataset(num_samples=100, input_dim=10, num_classes=3)
        print(f"✅ 数据加载测试通过 (训练批次: {len(train_loader)}, 测试批次: {len(test_loader)})")
        
        # 测试早停机制
        early_stopping = EarlyStopping(patience=3, verbose=False)
        for loss in [0.5, 0.4, 0.45, 0.46]:
            early_stopping(loss)
        print("✅ 早停机制测试通过")
        
        # 测试指标追踪
        tracker = MetricTracker()
        tracker.update(0.5, 85.0, 0.4, 87.0)
        tracker.update(0.3, 90.0, 0.35, 88.5)
        best_epoch, best_acc = tracker.get_best_epoch()
        assert best_epoch is not None, "指标追踪错误"
        print("✅ 指标追踪测试通过")
        
        return True
    except Exception as e:
        print(f"❌ 工具测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """测试完整的训练流程"""
    print("\n" + "=" * 60)
    print("测试完整训练流程")
    print("=" * 60)
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        from models import SimpleNN
        from utils import create_synthetic_dataset, train_epoch, validate, MetricTracker
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建数据
        train_loader, test_loader = create_synthetic_dataset(num_samples=200, input_dim=20, num_classes=3)
        
        # 创建模型
        model = SimpleNN(input_size=20, hidden_sizes=[32], num_classes=3).to(device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # 训练 2 个 epoch
        tracker = MetricTracker()
        for epoch in range(2):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, verbose=False)
            val_loss, val_acc = validate(model, test_loader, criterion, device, verbose=False)
            tracker.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f"✅ 训练流程测试通过 (最终准确率: {tracker.val_accs[-1]:.2f}%)")
        
        return True
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("PyTorch 多文件项目测试")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    # 运行所有测试
    results = []
    results.append(("模型模块", test_models()))
    results.append(("工具模块", test_utils()))
    results.append(("训练流程", test_training()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！项目可以正常使用。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

