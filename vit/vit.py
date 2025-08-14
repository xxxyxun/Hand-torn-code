import torch
import torch.nn as nn
from Block import Block
from patchembedlayer import PatchEmbedLayer

class ViT(nn.Module): 
    def __init__(self, img_size=224, 
                patch_size=16, 
                in_channels=3, 
                embed_dim=768, 
                nums_heads=12, 
                d_ff=3072, 
                num_layers=12,
                class_nums=1000,  # 最终分类数量 
                dropout=0.):
        super(ViT, self).__init__()

        # (batch_size, in_channels, img_size, img_size)
        # Patch embedding layer
        self.patch_embed = PatchEmbedLayer(img_size, patch_size, in_channels, embed_dim)
        
        # (batch_size, num_patches + 1, embed_dim)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, nums_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # (batch_size, num_patches + 1, embed_dim)
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, class_nums)  # 最终输出维度
        )

        # (batch_size, num_patches + 1, class_nums)
        self.softmax = nn.Softmax(dim=-1)  # 用于分类的softmax层


    def forward(self, x):
        # x: (batch_size, in_channels, img_size, img_size)
        x = self.patch_embed(x)  # (batch_size, num_patches + 1, embed_dim)
        
        for block in self.blocks:
            x = block(x)
        # (batch_size, num_patches + 1, embed_dim)
            

        output = x[:, 0, :]  # 取出分类token对应的输出 (batch_size, embed_dim)
        output = self.MLP_head(output)  # (batch_size, class_nums)
        output = self.softmax(output)
        
        return output  # 返回最后的输出
    
if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    nums_heads = 12
    d_ff = 3072
    num_layers = 12
    class_nums = 1000

    model = ViT(img_size, patch_size, in_channels, embed_dim, nums_heads, d_ff, num_layers, class_nums)

    
    x = torch.randn(2, in_channels, img_size, img_size)  # batch_size=2
    output = model(x)
    print("Output shape:", output.shape)  # 应该是 (batch_size, class_nums)

    #Output shape: torch.Size([2, 1000])