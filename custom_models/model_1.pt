import torch
import torch.nn as nn

class TimeSeriesModel(nn.Module):
    """
    示例时序预测模型
    这个模型展示了一个简单的时序预测模型结构，包含CNN和LSTM的组合
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # CNN层
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # 输入形状: [batch_size, sequence_length, input_dim]
        
        # CNN处理
        x = x.transpose(1, 2)  # [batch_size, input_dim, sequence_length]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)  # [batch_size, sequence_length/4, hidden_dim]
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        
        # 全连接层
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = TimeSeriesModel(
        input_dim=10,    # 输入特征维度
        hidden_dim=64,   # 隐藏层维度
        output_dim=1     # 输出维度
    )
    
    # 创建示例输入数据
    batch_size = 32
    sequence_length = 20
    x = torch.randn(batch_size, sequence_length, 10)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model) 