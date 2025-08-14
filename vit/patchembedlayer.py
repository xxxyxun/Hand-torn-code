import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedLayer(nn.Module):
    #这里参数按照原始的ViT论文来设置  
    #输入维度为(batch_size, in_channels, img_size, img_size)
    #输出维度为(batch_size, num_patches, embed_dim)
    def  __init__(self, img_size = 224, patch_size = 16, in_channels = 3, embed_dim = 768, norm_layer=None):
        super(PatchEmbedLayer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        #计算patch的数量
        num_patches = (img_size // patch_size) ** 2
        
        #定义卷积层来实现patch embedding
        self.layer = nn.Sequential(
            #维度(batch_size, embed_dim, img_size//patch_size, img_size//patch_size)
            #(b, 3, 224, 224) -> (b, 768, 14, 14)
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),  #将卷积输出展平为(batch_size, max_len, embed_dim)对应transformer的输入形状
        )
        '''
        self.layer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(in_channels * patch_size * patch_size, embed_dim)
            )
        '''

        #可选的归一化层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.postion_embedding = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        #初始化位置编码 对齐输入形状 batch_size维度上广播

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        #输入x的形状为(batch_size, in_channels, img_size, img_size)
        x = self.layer(x)
        #添加分类token
        cls_token = repeat(self.cls_token, '() 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_token, x), dim=1)
        #添加位置编码
        x = x + self.postion_embedding
        #如果有归一化层，则应用
        if self.norm is not None:
            x = self.norm(x)

        return x
    
#测试
if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768

    #创建一个随机输入张量 (batch_size, in_channels, img_size, img_size)
    batch_size = 2
    x = torch.randn(batch_size, in_channels, img_size, img_size)

    #创建PatchEmbedLayer实例
    patch_embed_layer = PatchEmbedLayer(img_size, patch_size, in_channels, embed_dim)

    #应用PatchEmbedLayer
    output = patch_embed_layer(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    # 输出形状应该是 (batch_size, num_patches + 1, embed_dim)
#Input shape: torch.Size([2, 3, 224, 224])
#Output shape: torch.Size([2, 197, 768])