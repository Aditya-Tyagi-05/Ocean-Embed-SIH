import torch
from ocean_embed.models.convlstm import ConvLSTM
from ocean_embed.models.vit import SpatiotemporalViT
x=torch.randn(2,7,16,16,2); y=torch.randn(2,16,16,2)
for name,model in [('ConvLSTM',ConvLSTM(hidden_channels=4)),('ViT',SpatiotemporalViT(height=16,width=16,patch_size=8,embedding_dim=16,heads=4,layers=1))]:
 opt=torch.optim.Adam(model.parameters(),lr=1e-3); pred=model(x); loss=((pred-y)**2).mean(); loss.backward(); opt.step(); torch.save({'model':model.state_dict()},f'/tmp/ocean_embed_{name.lower()}.pt'); print(name,tuple(pred.shape),float(loss))
