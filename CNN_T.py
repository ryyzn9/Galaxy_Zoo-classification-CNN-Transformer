import torch
from torch import nn 
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from torchvision import models


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class preNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)
    

class FFN(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0) -> None:
        super().__init__()
        self.L1= nn.Linear(dim,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self,x):
        x = self.L1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.L2(x)
        x = self.drop(x)
        return x    
    
class MHAttention(nn.Module):
    def __init__(self,dim , heads =8,dim_head=64,dropout=0.1):
        super().__init__()
        inner_dim = dim_head*heads
        self.heads = heads

        self.scale = dim_head ** -0.5
        
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(dropout)
        self.Q= nn.Linear(dim,inner_dim,bias=False)
        self.K= nn.Linear(dim,inner_dim,bias=False)
        self.V= nn.Linear(dim,inner_dim,bias=False)
        self.O =nn.Linear(inner_dim,dim,bias=False)

    def forward(self,x) :
           q = self.Q(x)
           k = self.K(x)
           v = self.V(x)
           
           q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
           k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
           v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)



           kq = torch.matmul(q,k.transpose(-1,-2))*self.scale #the n <--> d swap the  places
           attn = self.softmax(kq)
           out = torch.matmul(attn,v)

           out = rearrange(out,'b h n d -> b n (h d)')

           return self.O(out)
    

class Transformer(nn.Module):
    def __init__(self, dim,heads ,layers,dim_head,FFN_dim,dropout=0.01):
        super().__init__()

        L1 =[ nn.ModuleList([preNorm(dim,MHAttention(dim=dim,
                                                            heads=heads,
                                                            dim_head=dim_head,
                                                            dropout=dropout)),
                                    
                                    
                                    
                                    preNorm(dim,FFN(dim,FFN_dim,dropout=dropout))
                                    
                                    
                                    
                                    ])  for _ in range(layers) ]
        
        self.layers =nn.ModuleList(L1) 

    def forward(self,x):

        for attn,ffn in self.layers:
            x = attn(x)+x
            x = ffn(x)+x
        return x
    


class CNN(nn.Module):
    def __init__(self):
            super(CNN, self).__init__()
            # can tune
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                Rearrange('b c h w -> b (h w) c')
                # Rearrange('b c h w -> b c (h w)')
            )

    def forward(self, x):
             return self.cnn(x)


# class SimCNN(nn.Module):
#     def __init__(self):
#         super(SimCNN, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(8*8*8, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#
#     def forward(self, x):
#         return self.cnn(x)
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, pre_trained=False):
#         """
#         Args:
#             pre_trained: True if want to use pretrained weight else false
#         """
#         super(ResNet, self).__init__()
#         self.backbone = models.resnet34(pretrained=pre_trained)
#         self.reg = nn.Sequential(
#             # nn.Linear(2048, 1)
#             nn.Linear(512, 10)
#         )
#         self.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.backbone.fc = self.reg
#
#     def forward(self, x):
#         return self.backbone(x)


class CNNT(nn.Module):
    def __init__(self, num_classes, depth, heads, mlp_dim, pool='mean', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.to_patch_embedding = CNN()

        # hyper-params
        num_patches = 16 * 16
        dim = 32

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        # b: batch size  n: patch number  _: dim of patch
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    

    #########################################################################################################

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.cnn = nn.Sequential(

#           nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=2),
#           nn.BatchNorm2d(16),
#           nn.ReLU(),
#           nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2),
#           nn.BatchNorm2d(32),
#           nn.ReLU(),
#           Rearrange('b c h w ->b (h w) c')


#         )


# class CNNT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.to_patch_embedding = nn.Sequential(
#           nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=2),
#           nn.BatchNorm2d(16),
#           nn.ReLU(),
#           nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2),
#           nn.BatchNorm2d(32),
#           nn.ReLU(),
#         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head = nn.Linear(dim, num_classes)

#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape

#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x
#         return self.mlp_head(x)