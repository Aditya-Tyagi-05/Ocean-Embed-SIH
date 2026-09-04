"""Evaluate saved ConvLSTM and ViT checkpoints on the held-out test period."""
from pathlib import Path
import numpy as np, torch
from ocean_embed.preprocessing.netcdf_loader import load_wind, discover_wind_files
from ocean_embed.preprocessing.sequences import WindNormalizer, make_sequences
from ocean_embed.models.convlstm import ConvLSTM
from ocean_embed.models.vit import SpatiotemporalViT
from ocean_embed.training.metrics import evaluate

ROOT=Path(__file__).resolve().parents[1]
def main():
    files=discover_wind_files(ROOT/'dataset'); ds=load_wind(files); values=np.stack([ds[v].values for v in ('eastward_wind','northward_wind')],-1).astype('float32'); mask=np.isfinite(values); norm=WindNormalizer().fit(values,mask); z=norm.transform(np.where(mask,values,norm.mean)); X,Y,_,YM=make_sequences(z,mask); n=len(X); test_start=max(1,int(n*.85)); dev=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'); print('Wind files:',len(files),'Date range:',str(ds.time.values[0])[:10],'to',str(ds.time.values[-1])[:10]); print('Device:',dev,'Test sequences:',n-test_start)
    configs={'convlstm':ConvLSTM(hidden_channels=8,layers=2),'vit':SpatiotemporalViT(input_length=7,height=values.shape[1],width=values.shape[2],patch_size=8,embedding_dim=32,heads=4,layers=2)}
    for name,model in configs.items():
        path=ROOT/'checkpoints'/f'{name}_best.pt'
        if not path.exists(): print(f'{name}: checkpoint missing ({path})'); continue
        state=torch.load(path,map_location=dev,weights_only=False); model.load_state_dict(state['model_state']); model.to(dev).eval()
        with torch.no_grad(): pred=model(X[test_start:].to(dev)).cpu().numpy()
        result=evaluate(norm.inverse(pred),norm.inverse(Y.numpy()[test_start:]),YM.numpy()[test_start:]); print(f'\n{name} (epoch {state["epoch"]})'); [print(f'  {k}: {v}') for k,v in result.items()]
if __name__=='__main__': main()
