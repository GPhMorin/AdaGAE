from model import AdaGAE
import torch
import warnings
import numpy as np

import networkx as nx
from sklearn.cluster import OPTICS
from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm
import pickle
from data_loader import load_balsac

warnings.filterwarnings('ignore')

data = load_balsac()
labels = np.zeros(data.shape[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Device used: {device}")

indices = torch.LongTensor(np.vstack([data.row, data.col]))
values = torch.FloatTensor(data.data)
shape = torch.Size(data.shape)
X = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float16).to(device)

input_dim = data.shape[1]
layers = [input_dim, 1024, 64]
accs = [];
nmis = [];
emb = None

for lam in np.power(2.0, np.array(range(-10, 10, 2))):
    for neighbors in [5]:
        print('-----lambda={}, neighbors={}'.format(lam, neighbors))
        gae = AdaGAE(X, layers=layers, num_neighbors=neighbors, lam=lam, max_iter=50, max_epoch=10,
                     update=True, learning_rate=5*10**-3, inc_neighbors=5, device=device)
        emb = gae.run()
    # accs.append(acc)
    # nmis.append(nmi)
# print(accs)
# print(nmis)

labels = OPTICS().fit_predict(emb)
print(labels)