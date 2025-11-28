"""
训练工具函数
包含训练、验证、测试和模型保存等功能
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os


def train_epoch(model, train_loader, criterion, optimizer, device='cpu', verbose=True):
    """
    训练一个 epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        verbose: 是否显示进度条
    
    返回:
        平均损失, 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(train_loader, desc='Training') if verbose else train_loader
    
    for inputs, labels in iterator:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if verbose:
            iterator.set_postfix({
                'loss': running_loss / (total / inputs.size(0)),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device='cpu', verbose=True):
    """
    验证模型
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        verbose: 是否显示进度条
    
    返回:
        平均损失, 准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(val_loader, desc='Validation') if verbose else val_loader
    
    with torch.no_grad():
        for inputs, labels in iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if verbose:
                iterator.set_postfix({
                    'loss': running_loss / (total / inputs.size(0)),
                    'acc': 100. * correct / total
                })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def test(model, test_loader, device='cpu', verbose=True):
    """
    测试模型
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        verbose: 是否显示进度条
    
    返回:
        准确率, 预测结果, 真实标签
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    iterator = tqdm(test_loader, desc='Testing') if verbose else test_loader
    
    with torch.no_grad():
        for inputs, labels in iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if verbose:
                iterator.set_postfix({'acc': 100. * correct / total})
    
    test_acc = 100. * correct / total
    
    return test_acc, np.array(all_predictions), np.array(all_labels)


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """
    保存模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        loss: 当前损失
        acc: 当前准确率
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到: {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """
    加载模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
        device: 设备
    
    返回:
        epoch, loss, acc
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    
    print(f"从 epoch {epoch} 恢复训练")
    print(f"损失: {loss:.4f}, 准确率: {acc:.2f}%")
    
    return epoch, loss, acc


class EarlyStopping:
    """
    早停机制
    """
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        参数:
            patience: 容忍的 epoch 数
            min_delta: 最小改进量
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class MetricTracker:
    """
    指标追踪器
    """
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def update(self, train_loss, train_acc, val_loss=None, val_acc=None):
        """更新指标"""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accs.append(val_acc)
    
    def get_best_epoch(self):
        """获取最佳 epoch"""
        if len(self.val_accs) > 0:
            best_epoch = np.argmax(self.val_accs)
            return best_epoch, self.val_accs[best_epoch]
        return None, None
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "=" * 50)
        print("训练总结")
        print("=" * 50)
        print(f"总 epoch 数: {len(self.train_losses)}")
        print(f"最终训练损失: {self.train_losses[-1]:.4f}")
        print(f"最终训练准确率: {self.train_accs[-1]:.2f}%")
        
        if len(self.val_losses) > 0:
            print(f"最终验证损失: {self.val_losses[-1]:.4f}")
            print(f"最终验证准确率: {self.val_accs[-1]:.2f}%")
            best_epoch, best_acc = self.get_best_epoch()
            print(f"最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch + 1})")


if __name__ == "__main__":
    print("训练工具测试")
    
    # 测试早停机制
    print("\n测试早停机制:")
    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    losses = [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]
    for i, loss in enumerate(losses):
        print(f"Epoch {i+1}, Loss: {loss}")
        early_stopping(loss)
        if early_stopping.early_stop:
            print("触发早停!")
            break
    
    # 测试指标追踪
    print("\n测试指标追踪:")
    tracker = MetricTracker()
    tracker.update(0.5, 85.0, 0.4, 87.0)
    tracker.update(0.3, 90.0, 0.35, 88.5)
    tracker.update(0.2, 92.0, 0.3, 91.0)
    tracker.print_summary()

