import numpy as np, torch
from ocean_embed.preprocessing.sequences import make_sequences, WindNormalizer
from ocean_embed.models.convlstm import ConvLSTM
from ocean_embed.models.vit import SpatiotemporalViT
from ocean_embed.training.metrics import circular_error, evaluate

def test_sequences_and_normalizer():
 x=np.random.randn(20,4,4,2).astype('float32'); m=np.ones_like(x,dtype=bool); n=WindNormalizer().fit(x,m); z=n.transform(x); assert np.allclose(n.inverse(z),x); a,b,am,bm=make_sequences(z,m); assert a.shape==(13,7,4,4,2) and b.shape==(13,4,4,2)
def test_models():
 x=torch.randn(2,7,16,16,2); assert ConvLSTM(hidden_channels=4)(x).shape==(2,16,16,2); assert SpatiotemporalViT(height=16,width=16,patch_size=8,embedding_dim=16,heads=4,layers=1)(x).shape==(2,16,16,2)
def test_circular_metric(): assert np.allclose(circular_error(np.array([359]),np.array([1])),2)
