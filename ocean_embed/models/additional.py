import torch
from torch import nn
from .reconstruction import ConvLSTMReconstructor, ViTReconstructor

class TransformerReconstructor(nn.Module):
    def __init__(self,in_channels=7,out_channels=15,steps=7,height=68,width=80,embed=64,heads=4,layers=2):
        super().__init__(); self.h=height; self.w=width; self.proj=nn.Linear(in_channels,embed); self.pos=nn.Parameter(torch.zeros(1,steps,embed)); enc=nn.TransformerEncoderLayer(embed,heads,embed*4,batch_first=True); self.enc=nn.TransformerEncoder(enc,layers); self.head=nn.Linear(embed,out_channels)
    def forward(self,x):
        z=x.mean((2,3)); z=self.enc(self.proj(z)+self.pos[:,:z.shape[1]]); return self.head(z[:,-1])[:,:,None,None].expand(-1,-1,self.h,self.w).permute(0,2,3,1)

class ConvFormerReconstructor(nn.Module):
    def __init__(self,in_channels=7,out_channels=15,steps=7,height=68,width=80,embed=32,heads=4,layers=1):
        super().__init__(); self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(embed,heads,embed*4,batch_first=True),layers); self.conv=nn.Conv2d(in_channels,embed,3,padding=1); self.head=nn.Conv2d(embed,out_channels,1)
    def forward(self,x):
        b,t,h,w,c=x.shape; z=self.conv(x[:,-1].permute(0,3,1,2)); pooled=z.mean((2,3)).unsqueeze(1); z=z+self.temporal(pooled).squeeze(1)[:,:,None,None]; return self.head(torch.relu(z)).permute(0,2,3,1)

class UNet3DReconstructor(nn.Module):
    def __init__(self,in_channels=7,out_channels=15,**kwargs):
        super().__init__(); self.net=nn.Sequential(nn.Conv3d(in_channels,32,3,padding=1),nn.ReLU(),nn.Conv3d(32,32,3,padding=1),nn.ReLU(),nn.Conv3d(32,out_channels,1))
    def forward(self,x): return self.net(x.permute(0,4,1,2,3)).mean(2).permute(0,2,3,1)
