import torch
import torch.nn as nn

# 输入维度 (batch_size, seq_len, d_model)
# 位置编码维度 (seq_len, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 计算位置编码表 大小为 (max_len, d_model)
        self.pe =  torch.zeros(max_len, d_model)
        # 计算位置索引 为了计算大小为 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算公式中的分母 大小为 (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 计算位置编码 广播机制将div_term 扩展到 (1， d_model/2) 的形状
        # 偶数位置使用正弦函数 大小为 (max_len, d_model/2)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置使用余弦函数
        self.pe[:, 1::2] = torch.cos(position * div_term)

        # 添加一个维度 (1, max_len, d_model) 以便与输入张量相加
        self.pe = self.pe.unsqueeze(0)
        # 注册位置编码为一个缓冲区，这样它就不会被视为模型参数
        self.register_buffer('postion', self.pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len_ = x.size(1)
        x = x + self.pe[:, :seq_len_, :]
        return x


if __name__ == "__main__":
    # 测试
    d_model = 512
    positional_embedding = PositionalEncoding(d_model)
    
    batch_size = 2
    seq_len = 5
    # 创建一个随机输入张量 (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 应用位置编码
    output = positional_embedding(x)
    
    print("Positional Encoding:", positional_embedding.pe[:, :seq_len, :].shape)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    # 输出形状应该是 (batch_size, seq_len, d_model)
    # Input shape: torch.Size([2, 5, 512])
    # Output shape: torch.Size([2, 5, 512])