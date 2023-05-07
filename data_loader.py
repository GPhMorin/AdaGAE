import scipy.io as scio
import numpy as np

import pickle
import networkx as nx
from scipy.sparse import load_npz, csr_matrix
from tqdm import tqdm

UMIST = 'UMIST'
COIL20 = 'COIL20'
JAFFE = 'JAFFE'
PALM = 'Palm'
USPS = 'USPSdata_20_uni'
MNIST_TEST = 'mnist_test'
SEGMENT = 'segment_uni'
NEWS = '20news_uni'
TEXT = 'text1_uni'
ISOLET = 'Isolet'


def load_cora():
    path = 'data/cora.mat'
    data = scio.loadmat(path)
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['fea']
    X = X.astype(np.float32)
    X /= np.max(X)
    links = data['W']
    return X, labels, links


def load_data(name):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['X']
    X = X.astype(np.float32)
    X /= np.max(X)
    return X, labels


def load_demo():
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

    data = nx.to_scipy_sparse_matrix(G)
    return data


def build_array():
    dok = load_npz('../matrix.npz')
    csr = csr_matrix(dok)
    chunk_size = 100
    chunks = []
    for i in tqdm(range(0, csr.shape[0], chunk_size)):  # Prepare more than 100 GB of RAM
        chunks.append(csr[i:i+chunk_size,:].toarray())
    data = np.concatenate(chunks, axis=0)
    with open('matrix.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)


def load_balsac():
    data = load_npz('../matrix.npz')
    return data