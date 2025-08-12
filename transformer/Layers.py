import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super(FFN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 第一个线性变换 x: (batch_size, seq_len, d_ff)
            nn.ReLU(),                  # 激活函数
            nn.Linear(d_ff, d_model),   # 第二个线性变换 x: (batch_size, seq_len, d_model)
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.layer(x)
        return x
    

class Encoder_layer(nn.Module):
    def __init__(self, d_model, d_ff, nums_heads, dropout=0.):
        super(Encoder_layer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, nums_heads, dropout=dropout)
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.LayerNorm2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        # x: (batch_size, seq_len, d_model)
        attn_out = self.multi_head_attention(x, x, x, padding_mask)  
        out1 = self.LayerNorm1(x + attn_out)  # 残差连接和层归一化

        ffn_out = self.ffn(out1)
        out2 = self.LayerNorm2(out1 + ffn_out) 

        return out2  # (batch_size, seq_len, d_model)
    
class Decoder_layer(nn.Module):
    def __init__(self, d_model, d_ff, nums_heads, dropout=0.):
        super(Decoder_layer, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, nums_heads, dropout=dropout)
        self.LayerNorm1 = nn.LayerNorm(d_model)

        self.corss_attention = MultiHeadAttention(d_model, nums_heads, dropout=dropout)
        self.LayerNorm2 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff, dropout)
        self.LayerNorm3 = nn.LayerNorm(d_model)
        


    def forward(self, x, encoder_output, padding_mask=None, causal_mask=None):
        # x: (batch_size, seq_len, d_model)
        masked_attn_out = self.masked_attention(x, x, x, padding_mask)
        out1 = self.LayerNorm1(x + masked_attn_out)  # 残差连接和层归一化

        corss_attn_out = self.corss_attention(out1, encoder_output, encoder_output, causal_mask)
        out2 = self.LayerNorm2(out1 + corss_attn_out)

        ffn_out = self.ffn(out2)
        out3 = self.LayerNorm3(out2 + ffn_out) 

        return out3  # (batch_size, seq_len, d_model)
    
#测试
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048
    nums_heads = 8

    encoder = Encoder_layer(d_model, d_ff, nums_heads)
    decoder = Decoder_layer(d_model, d_ff, nums_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    padding_mask = None  # 假设没有padding_mask
    causal_mask = None   # 假设没有causal_mask

    encoder_output = encoder(x, padding_mask)
    decoder_output = decoder(x, encoder_output, padding_mask, causal_mask)

    print("Encoder output shape:", encoder_output.shape)
    print("Decoder output shape:", decoder_output.shape)

    # 应该是 (batch_size, seq_len, d_model)
    # Encoder output shape: torch.Size([2, 5, 512])
    # Decoder output shape: torch.Size([2, 5, 512])