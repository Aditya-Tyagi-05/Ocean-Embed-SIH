from __future__ import annotations
import numpy as np
import torch

class WindNormalizer:
    def __init__(self, mean=None, std=None): self.mean = np.asarray(mean if mean is not None else [0., 0.], dtype=np.float32); self.std = np.asarray(std if std is not None else [1., 1.], dtype=np.float32)
    def fit(self, values, mask=None):
        x = np.asarray(values, dtype=np.float32); self.mean = np.nanmean(np.where(mask, x, np.nan), axis=(0,1,2)); self.std = np.nanstd(np.where(mask, x, np.nan), axis=(0,1,2)); self.std = np.where(self.std < 1e-6, 1., self.std); return self
    def transform(self, x): return (x - self.mean) / self.std
    def inverse(self, x): return x * self.std + self.mean
    def state_dict(self): return {"mean": self.mean.tolist(), "std": self.std.tolist()}

def make_sequences(values, mask, input_length=7, horizon=1, stride=1):
    values = np.asarray(values, dtype=np.float32); mask = np.asarray(mask, dtype=bool); xs=[]; ys=[]; xm=[]; ym=[]
    for end in range(input_length, len(values)-horizon+1, stride):
        sl = slice(end-input_length, end); target = end+horizon-1
        xs.append(values[sl]); ys.append(values[target]); xm.append(mask[sl]); ym.append(mask[target])
    if not xs: raise ValueError("Not enough timesteps for requested sequence")
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(xm)), torch.from_numpy(np.stack(ym))

def chronological_split(values, train=.7, validation=.15):
    n=len(values); a=max(1,int(n*train)); b=max(a+1,int(n*(train+validation))); return values[:a], values[a:b], values[b:]
