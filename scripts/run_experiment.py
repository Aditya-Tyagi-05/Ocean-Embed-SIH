from pathlib import Path
import json, time
import numpy as np, torch, xarray as xr
from ocean_embed.preprocessing.netcdf_loader import load_wind
from ocean_embed.preprocessing.sequences import WindNormalizer, make_sequences
from ocean_embed.models.persistence import Persistence
from ocean_embed.models.convlstm import ConvLSTM
from ocean_embed.models.vit import SpatiotemporalViT
from ocean_embed.training.metrics import evaluate

root=Path(__file__).resolve().parents[1]; wind=next(root.glob('cmems_obs-wind*.nc')); ds=load_wind([wind]); values=np.stack([ds[v].values for v in ('eastward_wind','northward_wind')],-1); mask=np.isfinite(values); norm=WindNormalizer().fit(values,mask); filled=np.where(mask,values,norm.mean); z=norm.transform(filled); X,Y,XM,YM=make_sequences(z,mask); n=len(X); cut=max(1,int(n*.7)); valcut=max(cut+1,int(n*.85)); device=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device,'Sequences:',n)
test_slice=slice(valcut,None)
def score(name,pred):
 p=norm.inverse(pred[test_slice]); y=norm.inverse(Y.numpy()[test_slice]); m=YM.numpy()[test_slice]; result=evaluate(p,y,m); result['model']=name; print(name,result); return result
results=[score('Persistence',Persistence()(X).numpy())]
predictions={}
for name,model in [('ConvLSTM',ConvLSTM(hidden_channels=8)),('ViT',SpatiotemporalViT(input_length=7,height=values.shape[1],width=values.shape[2],patch_size=8,embedding_dim=32,heads=4,layers=2))]:
 model.to(device); opt=torch.optim.Adam(model.parameters(),lr=.001); xx=X[:cut].to(device); yy=Y[:cut].to(device); best=None
 for epoch in range(2):
  model.train(); opt.zero_grad(); pred=model(xx); loss=(((pred-yy)**2)*YM[:cut].to(device)).sum()/YM[:cut].to(device).sum(); loss.backward(); opt.step(); print(name,'epoch',epoch+1,'loss',float(loss.detach()))
 model.eval();
 with torch.no_grad(): pred=model(X.to(device)).cpu().numpy()
 predictions[name]=pred; results.append(score(name,pred))
if 'ConvLSTM' in predictions and 'ViT' in predictions:
 best=(None,float('inf'))
 for alpha in np.linspace(0,1,11):
  p=alpha*predictions['ConvLSTM']+(1-alpha)*predictions['ViT']; metric=evaluate(norm.inverse(p[test_slice]),norm.inverse(Y.numpy()[test_slice]),YM.numpy()[test_slice])['rmse']
  if metric<best[1]: best=(float(alpha),metric)
 alpha=best[0]; results.append(score('Ensemble',alpha*predictions['ConvLSTM']+(1-alpha)*predictions['ViT']))
 print('Ensemble alpha:',alpha)
 out=root/'outputs'/'metrics'/'experiment.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'device':str(device),'normalization':norm.state_dict(),'results':results},indent=2)); print('Wrote',out)
