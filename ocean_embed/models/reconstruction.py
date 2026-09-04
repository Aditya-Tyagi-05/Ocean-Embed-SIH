import torch
from torch import nn

class ConvLSTMReconstructor(nn.Module):
    def __init__(self,in_channels=7,out_channels=15,hidden=32):
        super().__init__(); self.cell=nn.Conv2d(in_channels+hidden,hidden,3,padding=1); self.head=nn.Conv2d(hidden,out_channels,1)
    def forward(self,x):
        b,t,h,w,c=x.shape; state=torch.zeros(b,self.cell.out_channels,h,w,device=x.device,dtype=x.dtype)
        for k in range(t): state=torch.relu(self.cell(torch.cat([x[:,k].permute(0,3,1,2),state],1)))
        return self.head(state).permute(0,2,3,1)

class ViTReconstructor(nn.Module):
    def __init__(self,in_channels=7,out_channels=15,steps=7,height=68,width=80,patch=8,embed=64,heads=4,layers=2):
        super().__init__(); self.h=height; self.w=width; self.patch=patch; self.ph=(height+patch-1)//patch; self.pw=(width+patch-1)//patch; self.out=out_channels; self.proj=nn.Conv3d(in_channels,embed,(1,patch,patch),stride=(1,patch,patch)); n=steps*self.ph*self.pw; self.pos=nn.Parameter(torch.zeros(1,n,embed)); enc=nn.TransformerEncoderLayer(embed,heads,embed*4,batch_first=True); self.enc=nn.TransformerEncoder(enc,layers); self.head=nn.Linear(embed,out_channels*patch*patch)
    def forward(self,x):
        b,t,h,w,c=x.shape; H=self.ph*self.patch; W=self.pw*self.patch; z=nn.functional.pad(x.permute(0,4,1,2,3),(0,W-w,0,H-h)); z=self.proj(z).permute(0,2,3,4,1).reshape(b,-1,self.pos.shape[-1]); z=self.enc(z+self.pos[:,:z.shape[1]]); z=self.head(z).reshape(b,t,self.ph,self.pw,self.out,self.patch,self.patch)[:,-1]; z=z.permute(0,3,1,4,2,5).reshape(b,self.out,H,W)[:,:,:h,:w]; return z.permute(0,2,3,1)
