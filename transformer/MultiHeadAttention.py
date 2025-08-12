import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nums_heads, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        #确定 d_model可以被 nums_heads 整除
        assert d_model % nums_heads == 0
        
        self.nums_heads = nums_heads
        self.d_model = d_model 
        self.d_k = d_model // nums_heads

        # 初始化Q, K, V投影层，将embedding词向量线性变换为 Q、K、V，维度保持一致
        self.linearW_q = nn.Linear(d_model, d_model)# = (d_model, d_k * nums_heads)
        self.linearW_k = nn.Linear(d_model, d_model)
        self.linearW_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        # 输出层
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attention_mask=None):
        # word_embedding: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = query.size()

        # 线性变换得到 Q、K、V 维度为 (batch_size, seq_len, d_model)
        Q = self.linearW_q(query)
        K = self.linearW_k(key)
        V = self.linearW_v(value)

        # 分离得到进行多头注意力计算的 Q、K、V 维度为 (batch_size, seq_len, nums_heads, d_k)
        Q = Q.view(batch_size, -1, self.nums_heads, self.d_k)
        K = K.view(batch_size, -1, self.nums_heads, self.d_k)
        V = V.view(batch_size, -1, self.nums_heads, self.d_k)

        # 转置以便进行多头注意力计算 nn.matmul是按最后两个维度进行矩阵乘法 (batch_size, nums_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力分数 维度(batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if attention_mask is not None:
            # 将注意力掩码应用于 scores
            scores = scores.masked_fill(attention_mask, float('-inf'))

        # 归一化注意力分数
        attention_probs = torch.softmax(scores, dim=-1)
        # 应用 dropout
        attention_probs = self.dropout(attention_probs)
        # 计算注意力输出 维度 (batch_size, num_heads, seq_len, d_k)
        attention_output = torch.matmul(attention_probs, V)
        # 将多头注意力输出合拼接 (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # 最后通过线性层输出
        output = self.linear_out(attention_output)

        return output
    
# 测试

if __name__ == "__main__":
    batch_size = 4
    seq_len = 5
    d_model = 8
    nums_heads = 4

    # 创建一个随机的输入张量
    input_tensor = torch.randn(batch_size, seq_len, d_model)

    # 创建 MultiHeadAttention 实例
    multiHeadAttention = MultiHeadAttention(d_model=d_model, nums_heads=nums_heads)

    # 前向传播
    output = multiHeadAttention(input_tensor, input_tensor, input_tensor)
    
    print("Output shape:", output.shape)  
    # 应该是 (batch_size, seq_len, d_model)
    # Output shape: torch.Size([4, 5, 8])
