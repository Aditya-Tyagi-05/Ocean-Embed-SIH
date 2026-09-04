import numpy as np
def masked_mae(p,y,m): return float(np.abs(p-y)[m].mean()) if m.any() else float('nan')
def masked_rmse(p,y,m): return float(np.sqrt(((p-y)**2)[m].mean())) if m.any() else float('nan')
def direction(u,v): return (np.degrees(np.arctan2(-u,-v))+360)%360
def circular_error(a,b): return np.abs((a-b+180)%360-180)
def evaluate(pred,target,mask):
    p=np.asarray(pred); y=np.asarray(target); m=np.asarray(mask); out={"u_mae":masked_mae(p[...,0],y[...,0],m[...,0]),"v_mae":masked_mae(p[...,1],y[...,1],m[...,1]),"rmse":masked_rmse(p,y,m)}
    ps=np.hypot(p[...,0],p[...,1]); ys=np.hypot(y[...,0],y[...,1]); sm=m.all(axis=-1); out["speed_mae"]=masked_mae(ps,ys,sm); out["direction_error"]=float(circular_error(direction(p[...,0],p[...,1]),direction(y[...,0],y[...,1]))[sm].mean()) if sm.any() else float('nan'); out["valid_pixels"]=int(m.sum()); out["valid_pct"]=float(m.mean()*100); return out
