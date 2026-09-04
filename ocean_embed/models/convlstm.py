import torch
from torch import nn

class ConvLSTM(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=32, layers=2, kernel_size=3, dropout=0.0):
        super().__init__(); self.blocks=nn.ModuleList(); c=input_channels
        for i in range(layers): self.blocks.append(nn.Conv2d(c+hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)); c=hidden_channels
        self.dropout=nn.Dropout2d(dropout); self.head=nn.Conv2d(hidden_channels, input_channels, 1)
    def forward(self, x):
        x=x.permute(0,1,4,2,3)
        states=[None]*len(self.blocks)
        for t in range(x.shape[1]):
            state=x[:,t]
            for i,layer in enumerate(self.blocks):
                if states[i] is None: states[i]=torch.zeros(state.shape[0],layer.out_channels,state.shape[2],state.shape[3],device=state.device,dtype=state.dtype)
                states[i]=torch.relu(layer(torch.cat([state,states[i]],dim=1))); state=states[i]
            state=self.dropout(state)
        return self.head(state).permute(0,2,3,1)
