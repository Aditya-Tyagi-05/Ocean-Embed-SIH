from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from ocean_embed.preprocessing.reconstruction_data import load_reconstruction_dataset
from ocean_embed.models.reconstruction import ConvLSTMReconstructor,ViTReconstructor
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['convlstm','vit'],required=True);p.add_argument('--data',default='dataset');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--seed',type=int,default=42);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 d=load_reconstruction_dataset(ROOT/a.data); x=torch.tensor(np.stack([d['inputs'][i-7:i] for i in range(7,len(d['inputs']))]));y=torch.tensor(d['targets'][7:]).permute(0,2,3,1);m=torch.tensor(d['target_mask'][7:]).permute(0,2,3,1); n=len(x); tr=max(1,int(n*.7));va=max(tr+1,int(n*.85)); train=DataLoader(TensorDataset(x[:tr],y[:tr],m[:tr]),a.batch_size,shuffle=False);valid=DataLoader(TensorDataset(x[tr:va],y[tr:va],m[tr:va]),a.batch_size);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'); model=ConvLSTMReconstructor(in_channels=x.shape[-1],out_channels=y.shape[-1]) if a.model=='convlstm' else ViTReconstructor(in_channels=x.shape[-1],out_channels=y.shape[-1],steps=x.shape[1],height=x.shape[2],width=x.shape[3]);model.to(dev);opt=torch.optim.Adam(model.parameters(),lr=a.lr);best=float('inf');print(f"Model: {a.model}\nDates: {d['dates'][0]} to {d['dates'][-1]}\nInput channels: {d['channels']}\nOutput depths: {d['depths'].tolist()}\nSequences: {n} (train={tr}, validation={va-tr}, test={n-va})\nDevice: {dev}")
 for epoch in range(1,a.epochs+1):
  t=time.perf_counter();model.train();total=0
  for xb,yb,mb in train: xb,yb,mb=xb.to(dev),yb.to(dev),mb.to(dev);opt.zero_grad();pred=model(xb);loss=(((pred-yb)**2)*mb).sum()/mb.sum().clamp_min(1);loss.backward();opt.step();total+=loss.item()*len(xb)
  model.eval();vt=0
  with torch.no_grad():
   for xb,yb,mb in valid: pred=model(xb.to(dev));vt+=((((pred-yb.to(dev))**2)*mb.to(dev)).sum()/mb.to(dev).sum().clamp_min(1)).item()*len(xb)
  tl=total/max(1,len(train.dataset));vl=vt/max(1,len(valid.dataset));print(f'Epoch {epoch:02d}/{a.epochs} | Train Loss: {tl:.5f} | Val Loss: {vl:.5f} | Time: {time.perf_counter()-t:.1f}s')
  if vl<best: best=vl;Path(ROOT/'checkpoints').mkdir(exist_ok=True);torch.save({'model':a.model,'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'depths':d['depths'].tolist(),'channels':d['channels'],'dates':d['dates'].tolist(),'device':str(dev)},ROOT/'checkpoints'/f'{a.model}_reconstruction_best.pt')
 print('Checkpoint saved.')
if __name__=='__main__':main()
