from pathlib import Path
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ocean_embed.preprocessing.netcdf_loader import inspect_file, load_wind, discover_wind_files
from ocean_embed.preprocessing.sequences import WindNormalizer
from ocean_embed.models.convlstm import ConvLSTM
from ocean_embed.models.vit import SpatiotemporalViT
from ocean_embed.training.metrics import direction
from ocean_embed.preprocessing.reconstruction_data import load_reconstruction_dataset
from ocean_embed.models.reconstruction import ConvLSTMReconstructor, ViTReconstructor

app=FastAPI(title="Ocean Embed API",version="0.2.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
ROOT=Path(__file__).resolve().parents[2]
DEVICE=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
class PredictionRequest(BaseModel): dataset: str="wind"; model: str="ensemble"; date: str|None=None
def wind_data():
    files=discover_wind_files(ROOT/"dataset"); ds=load_wind(files); values=np.stack([ds[v].values for v in ("eastward_wind","northward_wind")],-1).astype("float32"); mask=np.isfinite(values); norm=WindNormalizer().fit(values,mask); return ds,values,mask,norm
def model_prediction(model_name,x,h,w):
    def one(name):
        path=ROOT/"checkpoints"/f"{name}_best.pt"
        if not path.exists(): raise HTTPException(503,f"Missing checkpoint: {path.name}")
        model=ConvLSTM(hidden_channels=8,layers=2) if name=="convlstm" else SpatiotemporalViT(input_length=7,height=h,width=w,patch_size=8,embedding_dim=32,heads=4,layers=2)
        state=torch.load(path,map_location=DEVICE,weights_only=False); model.load_state_dict(state["model_state"]); model.to(DEVICE).eval()
        with torch.no_grad(): return model(torch.from_numpy(x[None]).to(DEVICE)).cpu().numpy()[0]
    if model_name=="persistence": return x[-1]
    if model_name=="ensemble": return .5*one("convlstm")+.5*one("vit")
    if model_name in {"convlstm","vit"}: return one(model_name)
    raise HTTPException(400,"Unknown model")
@app.get("/health")
def health(): return {"status":"ok","service":"ocean-embed","device":str(DEVICE)}
@app.get("/datasets")
def datasets(): return {"datasets":[inspect_file(p) for p in discover_wind_files(ROOT/"dataset")]}
@app.get("/metrics")
def metrics():
    p=ROOT/"outputs/metrics/experiment.json"; return {"results":json.loads(p.read_text()).get("results",[])} if p.exists() else {"results":[]}
@app.post("/predict")
def predict(req: PredictionRequest):
    ds,values,mask,norm=wind_data(); dates=[str(x)[:10] for x in ds.time.values]; target_date=req.date or dates[-1]
    if target_date not in dates: raise HTTPException(400,"Requested date is unavailable")
    target=dates.index(target_date)
    if target<7: raise HTTPException(400,"At least 7 previous observations are required")
    x=np.where(mask[target-7:target],values[target-7:target],norm.mean); x=norm.transform(x); pred=norm.inverse(model_prediction(req.model,x,values.shape[1],values.shape[2])); observed=values[target]; valid=mask[target] & np.isfinite(pred); speed=np.hypot(pred[...,0],pred[...,1]); obs_speed=np.hypot(observed[...,0],observed[...,1]); result={"model":req.model,"forecast_time":str(ds.time.values[target]),"latitude":ds.latitude.values.tolist(),"longitude":ds.longitude.values.tolist(),"u":pred[...,0].tolist(),"v":pred[...,1].tolist(),"wind_speed":speed.tolist(),"wind_direction":direction(pred[...,0],pred[...,1]).tolist(),"observed_speed":np.nan_to_num(obs_speed).tolist(),"error":np.nan_to_num(np.abs(speed-obs_speed)).tolist(),"valid_pct":float(valid.mean()*100),"device":str(DEVICE)}
    return result

@app.get("/reconstruct")
def reconstruct(date: str|None=None, model: str="ensemble", depth: float=150):
    d=load_reconstruction_dataset(ROOT/"dataset"); dates=d["dates"].tolist(); chosen=date or dates[-1]
    if chosen not in dates: raise HTTPException(400,"Requested date is unavailable")
    ti=dates.index(chosen)
    if ti<7: raise HTTPException(400,"At least 7 previous days are required")
    x=torch.tensor(d["inputs"][ti-7:ti][None],dtype=torch.float32).to(DEVICE); h,w=d["inputs"].shape[1:3]
    def run(name):
        path=ROOT/"checkpoints"/f"{name}_reconstruction_best.pt"
        if not path.exists(): raise HTTPException(503,f"Missing reconstruction checkpoint: {path.name}")
        m=ConvLSTMReconstructor(in_channels=7,out_channels=15) if name=="convlstm" else ViTReconstructor(in_channels=7,out_channels=15,steps=7,height=h,width=w)
        m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False)["model_state"]);m.to(DEVICE).eval()
        with torch.no_grad(): return m(x).cpu().numpy()[0]
    pred=d["targets"][ti].transpose(1,2,0) if model=="persistence" else .5*run("convlstm")+.5*run("vit") if model=="ensemble" else run(model)
    di=int(np.abs(d["depths"]-depth).argmin()); target=d["targets"][ti].transpose(1,2,0); err=np.abs(pred-target); return {"model":model,"date":chosen,"dates":dates,"depth":float(d["depths"][di]),"depths":d["depths"].tolist(),"latitude":d["latitude"].tolist(),"longitude":d["longitude"].tolist(),"temperature":pred[:,:,di].tolist(),"profile":np.nanmean(pred,axis=(0,1)).tolist(),"observed_temperature":target[:,:,di].tolist(),"observed_profile":np.nanmean(target,axis=(0,1)).tolist(),"error":err[:,:,di].tolist(),"units":"°C","input_variables":list(d["channels"]),"status":"January 2019 Bay of Bengal proof-of-concept"}

@app.get("/subsurface")
def subsurface(date: str|None=None, depth: float=150):
    files=sorted((ROOT/"dataset").rglob("cmems_mod_glo_phy_my_0.083deg_P1D-m_*.nc"))
    temp=[p for p in files if "thetao" in inspect_file(p)["variables"]]
    if not temp: raise HTTPException(404,"No subsurface temperature dataset found")
    with __import__('xarray').open_dataset(temp[0]) as ds:
        dates=[str(x)[:10] for x in ds.time.values]; chosen=date or dates[-1]
        if chosen not in dates: raise HTTPException(400,"Requested date is unavailable")
        ti=dates.index(chosen); depths=ds.depth.values.astype(float); di=int(np.abs(depths-depth).argmin()); field=ds.thetao.isel(time=ti,depth=di).values
        profile=ds.thetao.isel(time=ti).mean(dim=("latitude","longitude"),skipna=True).values
        return {"date":chosen,"depth":float(depths[di]),"depths":depths.tolist(),"profile":np.nan_to_num(profile,nan=0).tolist(),"latitude":ds.latitude.values.tolist(),"longitude":ds.longitude.values.tolist(),"temperature":np.nan_to_num(field,nan=0).tolist(),"units":"°C","source":temp[0].name,"status":"observed data explorer; no subsurface model trained"}
