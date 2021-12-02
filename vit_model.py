
import torch
import torch.nn as nn

class Patch_embeded(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, in_channel=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channel  = in_channel
        self.image_size = image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(self.in_channel, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, self.embed_dim))
    
    def forward(self, x):
        # x: [B, C, H, W]

        x = self.proj(x)
        x = x.flatten(2) # [B, embed_dim, num_patches]
        x = x.transpose(1, 2) # [B, num_patches, embed_dim]

        cls_token = self.cls_token.expand(x.shape[0], -1 ,-1)
        x = torch.cat((x,cls_token), dim=1)  #[B, num_patches+1, embed_dim]
        pos_embed = self.pos_embed

        out = x + pos_embed #[B, num_patches+1, embed_dim]
        return out

class attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False):
        super().__init__()

        self.num_heads = num_heads
        prehead_dim = dim // self.num_heads
        self.scale = prehead_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, num_patches+1, embed_dim]
        B, num_patches, total_dim = x.shape

        qkv = self.qkv(x) #[B, num_patches+1, 3*embed_dim]

        qkv = qkv.reshape(B, num_patches, 3, self.num_heads, total_dim // self.num_heads) #[B, num_patches+1, 3, num_heads, prehead_dim]

        qkv = qkv.permute(2, 0, 3, 1, 4) #[3, B, num_heads, num_patches+1, prehead_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  #[B, num_heads, num_patches+1, prehead_dim]

        atten = (q @ k.transpose(-2,-1)) * self.scale  #[B, num_heads, num_patches+1, num_patches+1]
        atten = atten.softmax(dim=-1)
        atten = atten @ v  ## [B, num_heads, num_patches+1, prehead_dim]
        atten = atten.transpose(1,2) #[B, num_patches+1, num_heads, prehead_dim]
        atten = atten.reshape(B, num_patches, total_dim)  #[B, num_patches+1, embed_dim]

        out = self.proj(atten)

        return out

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.actlayer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)  #[B, num_patches+1, hidden_dim]
        x = self.actlayer(x)
        x = self.fc2(x)  #[B, num_patches+1, out_dim]
        x = self.actlayer(x)

        return x

class Encoder_block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads, 
                 mlp_ration=4,
                 qkv_bias = False,

                 ):
        super().__init__()

        self.normlayer = nn.LayerNorm(dim)
        self.atten = attention(dim, num_heads, qkv_bias=qkv_bias)
        self.hidden_dim = int(dim*mlp_ration)
        self.mlp = MLP(in_dim = dim, hidden_dim = self.hidden_dim, out_dim=dim)
    
    def forward(self, x):
        x = x + self.atten(self.normlayer(x))
        x = x + self.mlp(self.normlayer(x))

        return x

class Vision_Model(nn.Module):
    def __init__(self,
                in_channel = 3,
                 dim=768,
                 num_heads = 12,
                 image_size = 224,
                 patch_szie = 16,
                 num_classes = 10,
                 depth = 12,
                 qkv_bias = True
                 ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_szie
        self.patch_embed = Patch_embeded(image_size= self.image_size, patch_size=self.patch_size, embed_dim=dim, in_channel=in_channel)
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        self.encoder = nn.Sequential(*[
            Encoder_block(dim=dim, num_heads= num_heads, mlp_ration=4, qkv_bias=qkv_bias) for i in range(depth)
        ])

        self.head = nn.Linear(dim, num_classes)

    def forward(self,x):

        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.head(x)
        return x


def main():
    input = torch.randn((10, 3, 224, 224))
    model = Vision_Model(in_channel = 3,
                 dim=768,
                 num_heads = 12,
                 image_size = 224,
                 patch_szie = 16,
                 num_classes = 10,
                 depth = 12,
                 qkv_bias = True)
    print(model)
    out = model(input)
    print(out[:,0]) ##output class token



if __name__ == '__main__':
    main()