from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import xarray as xr

WIND_VARS = ("eastward_wind", "northward_wind")

def discover_wind_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*.nc")):
        try:
            with xr.open_dataset(path) as ds:
                if all(v in ds.data_vars for v in WIND_VARS): files.append(path)
        except Exception:
            pass
    if not files: raise FileNotFoundError(f"No wind NetCDF files found below {root}")
    return files

def _coord(ds, names):
    for name in names:
        if name in ds.coords or name in ds:
            return name
    return None

def inspect_file(path: str | Path) -> dict:
    path = Path(path)
    with xr.open_dataset(path) as ds:
        time = _coord(ds, ("time",)); lat = _coord(ds, ("latitude", "lat")); lon = _coord(ds, ("longitude", "lon"))
        variables = {}
        for name, da in ds.data_vars.items():
            values = np.asarray(da.values, dtype=np.float64)
            finite = np.isfinite(values)
            variables[name] = {"dims": list(da.dims), "shape": list(da.shape), "units": da.attrs.get("units", ""),
                               "min": float(np.nanmin(values)) if finite.any() else None,
                               "max": float(np.nanmax(values)) if finite.any() else None,
                               "mean": float(np.nanmean(values)) if finite.any() else None,
                               "missing_pct": float((~finite).mean() * 100)}
        def info(name):
            if not name: return None
            a = ds[name].values
            return {"first": str(a.flat[0]), "last": str(a.flat[-1]), "size": int(a.size), "units": ds[name].attrs.get("units", "")}
        return {"filename": path.name, "path": str(path), "dimensions": {k: int(v) for k, v in ds.sizes.items()},
                "coordinates": {"time": info(time), "latitude": info(lat), "longitude": info(lon)},
                "variables": variables, "title": ds.attrs.get("title", "")}

def load_wind(paths: list[str | Path]) -> xr.Dataset:
    datasets = []
    for path in paths:
        ds = xr.open_dataset(path)
        missing = [v for v in WIND_VARS if v not in ds]
        if missing: ds.close(); raise ValueError(f"{path} missing required variables: {missing}")
        if "latitude" not in ds.coords or "longitude" not in ds.coords or "time" not in ds.coords:
            ds.close(); raise ValueError(f"{path} must contain time/latitude/longitude coordinates")
        datasets.append(ds[[*WIND_VARS]])
    reference = datasets[0]
    for ds in datasets[1:]:
        for coord in ("latitude", "longitude"):
            if not np.allclose(reference[coord].values, ds[coord].values, equal_nan=True):
                raise ValueError(f"Incompatible {coord} coordinate; regridding must be explicit")
        for var in WIND_VARS:
            if reference[var].attrs.get("units") != ds[var].attrs.get("units"):
                raise ValueError(f"Incompatible units for {var}")
    result = xr.concat(datasets, dim="time").sortby("time")
    _, unique = np.unique(result.time.values, return_index=True)
    return result.isel(time=np.sort(unique))

def write_registry(paths, output: str | Path):
    records = [inspect_file(p) for p in paths]
    wind = [r["filename"] for r in records if all(v in r["variables"] for v in WIND_VARS)]
    for r in records: r["compatible_for_wind_training"] = r["filename"] in wind
    Path(output).write_text(json.dumps({"files": records}, indent=2), encoding="utf-8")
    return records
