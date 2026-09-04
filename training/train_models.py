"""Train ConvLSTM and spatiotemporal ViT on the available ASCAT wind data.

Example:
    PYTHONPATH=. python3 training/train_models.py --epochs 5
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ocean_embed.preprocessing.netcdf_loader import load_wind, discover_wind_files
from ocean_embed.preprocessing.sequences import WindNormalizer, make_sequences
from ocean_embed.models.convlstm import ConvLSTM
from ocean_embed.models.vit import SpatiotemporalViT

ROOT = Path(__file__).resolve().parents[1]

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def masked_loss(pred, target, mask):
    mask = mask.float()
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Wind file or dataset root containing monthly folders")
    ap.add_argument("--model", choices=["convlstm", "vit", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data_root=Path(args.data) if args.data else ROOT/"dataset"
    wind_files=discover_wind_files(data_root) if data_root.is_dir() else [data_root]
    ds=load_wind(wind_files)
    values=np.stack([ds[v].values for v in ("eastward_wind","northward_wind")], axis=-1).astype("float32")
    valid=np.isfinite(values)
    normalizer=WindNormalizer().fit(values, valid)
    # Impute only model inputs; retain valid masks for loss and evaluation.
    normalized=normalizer.transform(np.where(valid, values, normalizer.mean))
    X,Y,XM,YM=make_sequences(normalized, valid)
    n=len(X); train_end=max(1,int(n*.70)); val_end=max(train_end+1,int(n*.85))
    train=DataLoader(TensorDataset(X[:train_end],Y[:train_end],YM[:train_end]),batch_size=args.batch_size,shuffle=False)
    val=DataLoader(TensorDataset(X[train_end:val_end],Y[train_end:val_end],YM[train_end:val_end]),batch_size=args.batch_size,shuffle=False)
    dev=device(); print(f"Wind files: {len(wind_files)}\nDate range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}\nSequences: {n} (train={train_end}, validation={val_end-train_end}, test={n-val_end})\nDevice: {dev}")
    models={"convlstm":ConvLSTM(hidden_channels=8,layers=2),"vit":SpatiotemporalViT(input_length=7,height=values.shape[1],width=values.shape[2],patch_size=8,embedding_dim=32,heads=4,layers=2)}
    models={k:v for k,v in models.items() if args.model in (k,"both")}
    ckpt_dir=ROOT/"checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    common={"dataset_root":str(data_root),"source_files":[str(p) for p in wind_files],"normalization":normalizer.state_dict(),"input_length":7,"forecast_horizon":1,"seed":args.seed,"device":str(dev),"grid":[int(values.shape[1]),int(values.shape[2])]}
    for name,model in models.items():
        model.to(dev); opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate); best=float("inf")
        for epoch in range(1,args.epochs+1):
            started=time.perf_counter(); model.train(); total=0.0
            for xb,yb,mb in train:
                xb,yb,mb=xb.to(dev),yb.to(dev),mb.to(dev); opt.zero_grad(); loss=masked_loss(model(xb),yb,mb); loss.backward(); opt.step(); total+=loss.item()*len(xb)
            model.eval(); val_total=0.0
            with torch.no_grad():
                for xb,yb,mb in val: val_total+=masked_loss(model(xb.to(dev),),yb.to(dev),mb.to(dev)).item()*len(xb)
            train_loss=total/max(len(train.dataset),1); val_loss=val_total/max(len(val.dataset),1)
            print(f"{name} | Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Time: {time.perf_counter()-started:.1f}s")
            if val_loss<best:
                best=val_loss; torch.save({**common,"model":name,"epoch":epoch,"validation_loss":val_loss,"model_config":model.__class__.__name__,"model_state":model.state_dict(),"optimizer_state":opt.state_dict()},ckpt_dir/f"{name}_best.pt")
    (ckpt_dir/"training_metadata.json").write_text(json.dumps({**common,"epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.learning_rate},indent=2))
    print(f"Saved checkpoints to {ckpt_dir}")

if __name__ == "__main__": main()
