"""Fit a PCA baseline that reconstructs temperature profiles from surface inputs."""
from pathlib import Path
import argparse,json,numpy as np
from sklearn.decomposition import PCA
from ocean_embed.preprocessing.reconstruction_data import load_reconstruction_dataset
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='dataset');p.add_argument('--components',type=int,default=8);a=p.parse_args();d=load_reconstruction_dataset(ROOT/a.data); y=d['targets'].transpose(0,2,3,1).reshape(len(d['targets']),-1,15); profiles=np.nanmean(y,axis=1); pca=PCA(n_components=min(a.components,15)).fit(profiles); out=ROOT/'checkpoints'/'pca_profile_best.npz';np.savez(out,components=pca.components_,mean=pca.mean_,explained=pca.explained_variance_ratio_,depths=d['depths']);(out.with_suffix('.json')).write_text(json.dumps({'model':'pca_profile','components':int(pca.n_components_),'dates':[str(d['dates'][0]),str(d['dates'][-1])]},indent=2));print('Saved:',out,'Explained variance:',float(pca.explained_variance_ratio_.sum()))
if __name__=='__main__':main()
