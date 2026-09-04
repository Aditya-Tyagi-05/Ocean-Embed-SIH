from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

STANDARD_DEPTHS = np.array([0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000], dtype="float32")
SURFACE_CHANNELS = ("sst", "sss", "sla", "current_u", "current_v", "wind_u", "wind_v")

def _one(folder, pattern):
    found=sorted(Path(folder).glob(pattern))
    if not found: raise FileNotFoundError(f"Missing {pattern} in {folder}")
    return found[0]

def _region(da, lat, lon):
    if "lat" in da.dims: da=da.rename({"lat":"latitude"})
    if "lon" in da.dims: da=da.rename({"lon":"longitude"})
    if float(da.longitude.max())>180: da=da.assign_coords(longitude=((da.longitude+180)%360)-180).sortby("longitude"); lon=((lon+180)%360)-180
    return da.interp(latitude=lat,longitude=lon,method="linear")

def load_reconstruction_month(folder: str|Path):
    folder=Path(folder); wind=xr.open_dataset(_one(folder,"*wind*.nc")); sla=xr.open_dataset(_one(folder,"*ssh*.nc")); curr=xr.open_dataset(_one(folder,"*0.083deg*")); theta=xr.open_dataset(sorted(folder.glob("*0.083deg*"))[1] if "thetao" not in curr else sorted(folder.glob("*0.083deg*"))[0]); sss=xr.open_dataset(_one(folder,"*sss*.nc"));
    # Wind/SLA define the final 0.25-degree Bay of Bengal grid.
    lat=wind.latitude.values; lon=wind.longitude.values; dates=np.array([str(t)[:10] for t in wind.time.values]);
    def date_slice(da, date): return da.sel(time=da.time.values[[str(t)[:10] for t in da.time.values].index(date)])
    sst_files=sorted(folder.glob("oisst*.nc")); sst_by={str(xr.open_dataset(p).time.values[0])[:10]:p for p in sst_files}; arrays=[]; targets=[]; masks=[]
    for date in dates:
        sst=xr.open_dataset(sst_by[date]).sst.isel(time=0,zlev=0) if date in sst_by else xr.full_like(wind.eastward_wind.isel(time=0),np.nan)
        sst=_region(sst,lat,lon); sss_d=_region(date_slice(sss["Sea_Surface_Salinity_Rain_Corrected"],date),lat,lon); sla_d=date_slice(sla.sla,date); cu=_region(date_slice(curr.uo,date).squeeze(),lat,lon); cv=_region(date_slice(curr.vo,date).squeeze(),lat,lon); wu=wind.eastward_wind.sel(time=date); wv=wind.northward_wind.sel(time=date)
        surface=np.stack([sst.values,sss_d.values,sla_d.values,cu.values,cv.values,wu.values,wv.values],-1).astype("float32"); arrays.append(surface); th=date_slice(theta.thetao,date).interp(latitude=lat,longitude=lon,depth=STANDARD_DEPTHS,method="linear",kwargs={"fill_value":"extrapolate"}).values.astype("float32"); targets.append(th); masks.append(np.isfinite(th))
    x=np.stack(arrays); y=np.stack(targets); target_mask=np.stack(masks); input_mask=np.isfinite(x); means=np.nanmean(x,axis=(0,1,2)); x=np.where(input_mask,x,means); y=np.where(target_mask,y,0)
    return {"dates":dates,"latitude":lat,"longitude":lon,"inputs":x,"input_mask":input_mask,"targets":y,"target_mask":target_mask,"channels":SURFACE_CHANNELS,"depths":STANDARD_DEPTHS}

def load_reconstruction_dataset(root: str|Path):
    """Load every complete monthly reconstruction folder below root."""
    root=Path(root); months=[]
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        try:
            months.append(load_reconstruction_month(folder))
            print(f"Loaded reconstruction month: {folder.name}")
        except FileNotFoundError as exc:
            print(f"Skipping {folder.name}: {exc}")
    if not months: raise FileNotFoundError(f"No complete reconstruction month folders found below {root}")
    ref=months[0]
    for item in months[1:]:
        if not np.allclose(ref["latitude"],item["latitude"]) or not np.allclose(ref["longitude"],item["longitude"]): raise ValueError("Monthly folders use different final grids")
        if tuple(item["channels"])!=tuple(ref["channels"]): raise ValueError("Monthly folders use different input channels")
    dates=np.concatenate([m["dates"] for m in months]); inputs=np.concatenate([m["inputs"] for m in months]); imask=np.concatenate([m["input_mask"] for m in months]); targets=np.concatenate([m["targets"] for m in months]); tmask=np.concatenate([m["target_mask"] for m in months]); order=np.argsort(dates); dates,inputs,imask,targets,tmask=dates[order],inputs[order],imask[order],targets[order],tmask[order]
    _,unique=np.unique(dates,return_index=True); keep=np.sort(unique)
    return {"dates":dates[keep],"latitude":ref["latitude"],"longitude":ref["longitude"],"inputs":inputs[keep],"input_mask":imask[keep],"targets":targets[keep],"target_mask":tmask[keep],"channels":ref["channels"],"depths":ref["depths"],"months":[m["dates"].tolist() for m in months]}
