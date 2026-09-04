"""Shared trainer for all neural reconstruction models."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from ocean_embed.preprocessing.reconstruction_data import load_reconstruction_dataset
from ocean_embed.models.reconstruction import ConvLSTMReconstructor,ViTReconstructor
from ocean_embed.models.additional import TransformerReconstructor,ConvFormerReconstructor,UNet3DReconstructor
ROOT=Path(__file__).resolve().parents[1]
MODELS=('convlstm','vit','spatiotemporal_vit','transformer','convformer','unet3d')
def make_model(name,channels,depths,h,w):
    kw=dict(in_channels=channels,out_channels=depths,steps=7,height=h,width=w)
    if name=='convlstm': return ConvLSTMReconstructor(**kw)
    if name in ('vit','spatiotemporal_vit'): return ViTReconstructor(**kw)
    if name=='transformer': return TransformerReconstructor(**kw)
    if name=='convformer': return ConvFormerReconstructor(**kw)
    return UNet3DReconstructor(in_channels=channels,out_channels=depths)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--data',default='dataset');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--patience',type=int,default=12);p.add_argument('--seed',type=int,default=42);p.add_argument('--no-resume',action='store_true');a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 d=load_reconstruction_dataset(ROOT/a.data); X=np.stack([d['inputs'][i-7:i] for i in range(7,len(d['inputs']))]);Y=d['targets'][7:].transpose(0,2,3,1);M=d['target_mask'][7:].transpose(0,2,3,1);x,y,m=map(torch.tensor,(X,Y,M));n=len(x);tr=max(1,int(n*.7));va=max(tr+1,int(n*.85));train=DataLoader(TensorDataset(x[:tr],y[:tr],m[:tr]),a.batch_size);val=DataLoader(TensorDataset(x[tr:va],y[tr:va],m[tr:va]),a.batch_size);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu');model=make_model(a.model,x.shape[-1],y.shape[-1],x.shape[2],x.shape[3]).to(dev);opt=torch.optim.Adam(model.parameters(),lr=a.lr);path=ROOT/'checkpoints'/f'{a.model}_reconstruction_best.pt';start=1;best=float('inf');bad=0
 if path.exists() and not a.no_resume:
  state=torch.load(path,map_location=dev,weights_only=False);model.load_state_dict(state['model_state']);opt.load_state_dict(state['optimizer_state']);start=state.get('epoch',0)+1;best=state.get('validation_loss',best);print(f'Resuming {a.model} from epoch {start}')
 print(f'Model: {a.model} | Dates: {d["dates"][0]} to {d["dates"][-1]} | Samples: {n} | Train/Val/Test: {tr}/{va-tr}/{n-va} | Device: {dev}')
 for epoch in range(start,a.epochs+1):
  t=time.perf_counter();model.train();total=0
  for xb,yb,mb in train: xb,yb,mb=xb.to(dev),yb.to(dev),mb.to(dev);opt.zero_grad();loss=(((model(xb)-yb)**2)*mb).sum()/mb.sum().clamp_min(1);loss.backward();opt.step();total+=loss.item()*len(xb)
  model.eval();vt=0
  with torch.no_grad():
   for xb,yb,mb in val: vt+=((((model(xb.to(dev))-yb.to(dev))**2)*mb.to(dev)).sum()/mb.to(dev).sum().clamp_min(1)).item()*len(xb)
  tl=total/max(1,len(train.dataset));vl=vt/max(1,len(val.dataset));print(f'Epoch {epoch:03d}/{a.epochs} | Train {tl:.5f} | Val {vl:.5f} | {time.perf_counter()-t:.1f}s')
  if vl<best: best=vl;bad=0;path.parent.mkdir(exist_ok=True);torch.save({'model':a.model,'epoch':epoch,'validation_loss':vl,'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'depths':d['depths'].tolist(),'channels':list(d['channels']),'dates':d['dates'].tolist(),'device':str(dev)},path)
  else: bad+=1
  if bad>=a.patience: print(f'Early stopping after {a.patience} epochs without validation improvement.');break
 print('Best checkpoint:',path)
if __name__=='__main__':main()
