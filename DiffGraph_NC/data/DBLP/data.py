# build_npz.py

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, save_npz, load_npz

# 1) Utility to load an edge‑list file into two lists of ints
def load_edges(fn):
    src, dst = [], []
    with open(fn, 'r') as f:
        for line in f:
            i, j = line.strip().split()
            src.append(int(i))
            dst.append(int(j))
    return src, dst

# 2) Read all six raw files
edges = {}
for name in ('ap','pa','cp','pc','pt','tp'):
    edges[name] = load_edges(f'{name}.txt')

# 3) Infer domain sizes by taking max + 1 over each src/dst
num_authors = max(edges['ap'][0] + edges['pa'][1]) + 1
num_papers  = max(edges['ap'][1] + edges['pa'][0] +
                  edges['cp'][1] + edges['pc'][0] +
                  edges['pt'][0] + edges['tp'][1]) + 1
num_confs   = max(edges['cp'][0] + edges['pc'][1]) + 1
num_terms   = max(edges['pt'][1] + edges['tp'][0]) + 1

# 4) Build and save single‑hop matrices (using variable names ap, pa, etc.)
ap = coo_matrix((np.ones(len(edges['ap'][0]),dtype=np.float32),
                 (edges['ap'][0], edges['ap'][1])),
                shape=(num_authors, num_papers))
save_npz('ap.npz', ap)

pa = coo_matrix((np.ones(len(edges['pa'][0]),dtype=np.float32),
                 (edges['pa'][0], edges['pa'][1])),
                shape=(num_papers, num_authors))
save_npz('pa.npz', pa)

cp = coo_matrix((np.ones(len(edges['cp'][0]),dtype=np.float32),
                 (edges['cp'][0], edges['cp'][1])),
                shape=(num_confs, num_papers))
save_npz('cp.npz', cp)

pc = coo_matrix((np.ones(len(edges['pc'][0]),dtype=np.float32),
                 (edges['pc'][0], edges['pc'][1])),
                shape=(num_papers, num_confs))
save_npz('pc.npz', pc)

pt = coo_matrix((np.ones(len(edges['pt'][0]),dtype=np.float32),
                 (edges['pt'][0], edges['pt'][1])),
                shape=(num_papers, num_terms))
save_npz('pt.npz', pt)

tp = coo_matrix((np.ones(len(edges['tp'][0]),dtype=np.float32),
                 (edges['tp'][0], edges['tp'][1])),
                shape=(num_terms, num_papers))
save_npz('tp.npz', tp)

print("Single‑hop matrices written: ap.npz, pa.npz, cp.npz, pc.npz, pt.npz, tp.npz")


# 6) Quick sanity check — load & print shape/nnz
for name in ['ap','pa','cp','pc','pt','tp']:
    M = load_npz(f'{name}.npz')
    print(f"{name}.npz → shape={M.shape}, nnz={M.nnz}")
