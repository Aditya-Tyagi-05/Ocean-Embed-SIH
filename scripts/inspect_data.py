from pathlib import Path
import json
from ocean_embed.preprocessing.netcdf_loader import inspect_file, write_registry

root=Path(__file__).resolve().parents[1]; files=sorted(root.glob('*.nc')); out=root/'outputs'/'dataset_registry.json'; records=write_registry(files,out)
for r in records:
 print(f"\nFILE: {r['filename']}\nDIMENSIONS: {r['dimensions']}")
 for n,c in r['coordinates'].items():
  if c: print(f"{n.upper()}: {c['first']} → {c['last']}")
 for n,v in r['variables'].items(): print(f"VARIABLE: {n} | dims={v['shape']} | units={v['units']} | range={v['min']}..{v['max']} | mean={v['mean']} | missing={v['missing_pct']:.2f}%")
print(f"\nRegistry written to {out}")
