import pandas as pd
import torch 



img_path = './datasets/image_run2d_dr16q.csv'
spec_path = './datasets/spec_run2d_dr16q.csv'
N = 10000

imgs = pd.read_csv(img_path, header=None)       
print(imgs.shape)
imgs = imgs.values[: N]                                     # (N, 5*24*24)
torch.save(torch.Tensor(imgs), './datasets/imgs_{}.pt'.format(N))
print(imgs.shape)
del imgs

specs = pd.read_csv(spec_path, header=None).values[: N]     # (N, 3600)
torch.save(torch.Tensor(specs), './datasets/specs_{}.pt'.format(N))
print(specs.shape)
del specs


