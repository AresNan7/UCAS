"""
LSTM 循环神经网络模型
适用于序列数据和时间序列预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    """
    简单的 LSTM 分类器
    适用于文本分类、序列分类等任务
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=10, dropout=0.5):
        """
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            num_classes: 输出类别数
            dropout: Dropout 比率
        """
        super(SimpleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, hidden=None):
        """
        前向传播
        x: (batch_size, seq_length)
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Dropout 和全连接
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        初始化隐藏状态
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class BidirectionalLSTM(nn.Module):
    """
    双向 LSTM 模型
    能够同时利用过去和未来的信息
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=10, dropout=0.5):
        """
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            num_classes: 输出类别数
            dropout: Dropout 比率
        """
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 双向 LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 注意力机制（可选）
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 全连接输出层（双向所以维度 x2）
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def attention_forward(self, lstm_out):
        """
        简单的注意力机制
        """
        # 计算注意力权重
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # 加权求和
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return context
        
    def forward(self, x, use_attention=True):
        """
        前向传播
        x: (batch_size, seq_length)
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # 双向 LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim*2)
        
        if use_attention:
            # 使用注意力机制
            context = self.attention_forward(lstm_out)
        else:
            # 取最后一个时间步的输出
            context = lstm_out[:, -1, :]
        
        # Dropout 和全连接
        out = self.dropout(context)
        out = self.fc(out)
        
        return out


class LSTMForSequenceGeneration(nn.Module):
    """
    用于序列生成的 LSTM 模型
    适用于文本生成、时间序列预测等任务
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(LSTMForSequenceGeneration, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        output = self.fc(lstm_out)
        
        return output, hidden


if __name__ == "__main__":
    # 测试 SimpleLSTM
    print("=" * 50)
    print("测试 SimpleLSTM")
    print("=" * 50)
    vocab_size = 10000
    model = SimpleLSTM(vocab_size, num_classes=5)
    
    batch_size = 16
    seq_length = 50
    test_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    output, hidden = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试 BidirectionalLSTM
    print("\n" + "=" * 50)
    print("测试 BidirectionalLSTM")
    print("=" * 50)
    model = BidirectionalLSTM(vocab_size, num_classes=5)
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")

