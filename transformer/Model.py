import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from Block import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, nums_heads, num_layers, dropout=0., max_len=5000):
        super(Transformer, self).__init__()
        # 这里dropout参数在所有层中相同
        self.Encoder_Embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.Dropout(dropout)
        )

        self.Decoder_Embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.Dropout(dropout)
        )

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder = Encoder(d_model, d_ff, nums_heads, num_layers, dropout)
        self.decoder = Decoder(d_model, d_ff, nums_heads, num_layers, dropout)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, causal_mask=None):
        
        # 生成填充掩码 这里默认生成
        src_padding_mask = self.generate_padding_mask(src, pad_idx=0) 
        tgt_padding_mask = self.generate_padding_mask(tgt, pad_idx=0)
        # 生成因果掩码
        causal_mask = self.generate_causal_mask(tgt.size(1))
        
        # src: (batch_size, src_seq_len) encoder输入
        # tgt: (batch_size, tgt_seq_len) decoder输入
        src = self.positional_encoding(self.Encoder_Embedding(src))
        tgt = self.positional_encoding(self.Decoder_Embedding(tgt))
        
        encder_output = self.encoder(src, src_padding_mask)
        decder_output = self.decoder(tgt, encder_output, tgt_padding_mask, causal_mask)

        output = self.output_layer(decder_output)
        return output
    
    def generate_causal_mask(self, tgt_seq_len):
        # 生成因果掩码，防止模型在训练时看到未来的词
        mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool()
        return mask.unsqueeze(0) # (1, tgt_seq_len, tgt_seq_len)
    
    def generate_padding_mask(self, seq, pad_idx=0):
        # 生成填充掩码，防止模型在计算注意力时关注到填充的部分
        return (seq == pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
    
# 测试
if __name__ == "__main__":
    vocab_size = 10000
    d_model = 512
    d_ff = 2048
    nums_heads = 8
    num_layers = 6
    dropout = 0.1
    max_len = 5000

    transformer = Transformer(vocab_size, d_model, d_ff, nums_heads, num_layers, dropout, max_len)

    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 10

    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    output = transformer(src, tgt)
    
    print("Output shape:", output.shape)  # 应该是 (batch_size, tgt_seq_len, vocab_size)
    # Output shape: torch.Size([2, 10, 10000])