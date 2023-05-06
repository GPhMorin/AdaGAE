from model import AdaGAE
import torch
import warnings
import numpy as np

import networkx as nx
from sklearn.cluster import OPTICS
from scipy.sparse import coo_matrix, load_npz
from sklearn.decomposition import PCA, TruncatedSVD

warnings.filterwarnings('ignore')

G = nx.MultiGraph()

G.add_node('fils A')
G.add_node('père A')
G.add_node('mère A')
G.add_node('grand-père paternel A')
G.add_node('grand-mère paternelle A')
G.add_node('grand-père maternel A')
G.add_node('grand-mère maternelle A')

G.add_node('fille B')
G.add_node('père B')
G.add_node('mère B')
G.add_node('grand-père paternel B')
G.add_node('grand-mère paternelle B')
G.add_node('grand-père maternel B')
G.add_node('grand-mère maternelle B')

G.add_edge('fils A', 'père A', weight=0.5)
G.add_edge('fils A', 'mère A', weight=0.5)
G.add_edge('fils A', 'grand-père paternel A', weight=0.25)
G.add_edge('fils A', 'grand-mère paternelle A', weight=0.25)
G.add_edge('fils A', 'grand-père maternel A', weight=0.25)
G.add_edge('fils A', 'grand-mère maternelle A', weight=0.25)

G.add_edge('père A', 'grand-père paternel A', weight=0.5)
G.add_edge('père A', 'grand-mère paternelle A', weight=0.5)
G.add_edge('mère A', 'grand-père maternel A', weight=0.5)
G.add_edge('mère A', 'grand-mère maternelle A', weight=0.5)

G.add_edge('fille B', 'père B', weight=0.5)
G.add_edge('fille B', 'mère B', weight=0.5)
G.add_edge('fille B', 'grand-père paternel B', weight=0.25)
G.add_edge('fille B', 'grand-mère paternelle B', weight=0.25)
G.add_edge('fille B', 'grand-père maternel B', weight=0.25)
G.add_edge('fille B', 'grand-mère maternelle B', weight=0.25)

G.add_edge('père B', 'grand-père paternel B', weight=0.5)
G.add_edge('père B', 'grand-mère paternelle B', weight=0.5)
G.add_edge('mère B', 'grand-père maternel B', weight=0.5)
G.add_edge('mère B', 'grand-mère maternelle B', weight=0.5)

data = nx.to_scipy_sparse_array(G)

data = load_npz('../../results/matrix.npz')
data = coo_matrix(data)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Device used: {device}")

X = torch.sparse_coo_tensor(
    indices=torch.LongTensor([data.row, data.col]),
    values=torch.FloatTensor(data.data),
    size=data.shape,
).to(device)

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

pca = OPTICS().fit_predict(PCA(n_components=2).fit_transform(np.asarray(data.todense())))
svd = OPTICS().fit_predict(TruncatedSVD(n_components=2).fit_transform(data))
adagae = OPTICS().fit_predict(emb)
print(f'PCA: {pca}')
print(f'TruncatedSVD: {svd}')
print(f'AdaGAE: {adagae}')