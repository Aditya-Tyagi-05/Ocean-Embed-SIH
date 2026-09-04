import torch
from torch import nn

class SpatiotemporalViT(nn.Module):
    def __init__(self, input_channels=2, input_length=7, height=68, width=80, patch_size=8, embedding_dim=64, heads=4, layers=2, dropout=.1):
        super().__init__(); self.h=height; self.w=width; self.ph=(height+patch_size-1)//patch_size; self.pw=(width+patch_size-1)//patch_size; self.ps=patch_size
        self.proj=nn.Conv3d(input_channels, embedding_dim, (1,patch_size,patch_size), stride=(1,patch_size,patch_size)); n=input_length*self.ph*self.pw
        self.pos=nn.Parameter(torch.zeros(1,n,embedding_dim)); enc=nn.TransformerEncoderLayer(embedding_dim,heads,embedding_dim*4,dropout,batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(enc,layers); self.head=nn.Linear(embedding_dim,input_channels*patch_size*patch_size); self.norm=nn.LayerNorm(embedding_dim)
    def forward(self,x):
        b,t,h,w,c=x.shape; ph=self.ph*self.ps; pw=self.pw*self.ps; x=x.permute(0,4,1,2,3)
        x=nn.functional.pad(x,(0,pw-w,0,ph-h)); z=self.proj(x).permute(0,2,3,4,1).reshape(b,-1,self.pos.shape[-1]); z=self.encoder(z+self.pos[:,:z.shape[1]]); z=self.head(self.norm(z)).reshape(b,t,self.ph,self.pw,c*self.ps*self.ps)[:,-1]
        z=z.reshape(b,self.ph,self.pw,c,self.ps,self.ps).permute(0,3,1,4,2,5).reshape(b,c,ph,pw)[:,:,:h,:w]; return z.permute(0,2,3,1)
