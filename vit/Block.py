import torch
import torch.nn as nn

# vit中只用了encoder部分
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, nums_heads, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        #确定 d_model可以被 nums_heads 整除
        assert embed_dim % nums_heads == 0
        
        self.nums_heads = nums_heads
        self.embed_dim = embed_dim
        self.d_k = embed_dim // nums_heads

        # 初始化Q, K, V投影层，将embedding词向量线性变换为 Q、K、V，维度保持一致
        self.linearW_q = nn.Linear(embed_dim, embed_dim)# = (embed_dim, d_k * nums_heads)
        self.linearW_k = nn.Linear(embed_dim, embed_dim)
        self.linearW_v = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        # 输出层
        self.linear_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attention_mask=None):
        # word_embedding: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = query.size()

        # 线性变换得到 Q、K、V 维度为 (batch_size, seq_len, embed_dim)
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
        # vit中没有使用mask，所以这里可以不加
        if attention_mask is not None:
            # 将注意力掩码应用于 scores
            scores = scores.masked_fill(attention_mask, float('-inf'))

        # 归一化注意力分数
        attention_probs = torch.softmax(scores, dim=-1)
        # 应用 dropout
        attention_probs = self.dropout(attention_probs)
        # 计算注意力输出 维度 (batch_size, num_heads, seq_len, d_k)
        attention_output = torch.matmul(attention_probs, V)
        # 将多头注意力输出合拼接 (batch_size, seq_len, embed_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # 最后通过线性层输出
        output = self.linear_out(attention_output)

        return output
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout=0.):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, d_ff),  # 第一层线性变换
            nn.GELU(),                  # 激活函数
            nn.Dropout(dropout),        # dropout
            nn.Linear(d_ff, embed_dim)    # 第二层线性变换
        )

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = self.layer(x)
        return x
    
class Block(nn.Module):
    def __init__(self, embed_dim, nums_heads, d_ff, dropout=0.):
        super(Block, self).__init__()

        self.Norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, nums_heads, dropout)
        self.dropout = nn.Dropout(dropout)

        # 前馈网络
        self.layer2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FeedForward(embed_dim, d_ff, dropout),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        # 1. 多头注意力
        x1 = self.Norm(x)
        x1 = self.attn(x1, x1, x1)
        x1 = self.dropout(x1)
        x = x + x1 

        # 2. 前馈网络
        ffn_out = self.layer2(x)
        x = x + ffn_out

        return x  # (batch_size, seq_len, embed_dim)
    
if __name__ == "__main__":
    embed_dim = 768
    nums_heads = 12
    d_ff = 3072
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    block = Block(embed_dim, nums_heads, d_ff, dropout)
    x = torch.randn(batch_size, seq_len, embed_dim)

    output = block(x)
    print("Output shape:", output.shape)  # 应该是 (batch_size, seq_len, embed_dim)