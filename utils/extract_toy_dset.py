import pandas as pd
import numpy as np

img_path = '../datasets/image_run2d_dr16q.csv'
N = 1000

imgs = pd.read_csv(img_path, header=None).values[: N]       # (N, 5*24*24)
print(imgs.shape)

np.save('imgs_1000.npy', imgs)

