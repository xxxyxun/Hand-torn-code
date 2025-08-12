import torch
import torch.nn as nn
from Layers import Encoder_layer, Decoder_layer

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, nums_heads, num_layers, dropout=0.):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            Encoder_layer(d_model, d_ff, nums_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None):
        # x: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x  # (batch_size, seq_len, d_model)
    
class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, nums_heads, num_layers, dropout=0.):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            Decoder_layer(d_model, d_ff, nums_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, padding_mask=None, causal_mask=None):
        # x: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, padding_mask, causal_mask)
        return x  # (batch_size, seq_len, d_model)
    
# 测试
if __name__ == "__main__":
    d_model = 512
    d_ff = 2048
    nums_heads = 8
    num_layers = 6
    dropout = 0.1

    encoder = Encoder(d_model, d_ff, nums_heads, num_layers, dropout)
    decoder = Decoder(d_model, d_ff, nums_heads, num_layers, dropout)

    batch_size = 2
    seq_len = 5

    x_encoder = torch.randn(batch_size, seq_len, d_model)
    x_decoder = torch.randn(batch_size, seq_len, d_model)

    output_encoder = encoder(x_encoder)
    output_decoder = decoder(x_decoder, x_encoder)

    print("Encoder output shape:", output_encoder.shape)  # 应该是 (batch_size, seq_len, d_model)
    print("Decoder output shape:", output_decoder.shape)  # 应该是 (batch_size, seq_len, d_model)
    # Encoder output shape: torch.Size([2, 5, 512])
    # Decoder output shape: torch.Size([2, 5, 512])