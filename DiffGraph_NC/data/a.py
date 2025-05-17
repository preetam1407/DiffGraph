import os
import numpy as np

# —— CONFIGURE THIS —— 
RAW_DIR   = "./dataDBLP/"    # your folder with ap.txt, pa.txt, etc.
OUT_DIR   = "./data/DBLP_small/"
MAX_ID    = 100               # keep only nodes numbered 0…MAX_ID
# —— end configuration —

os.makedirs(OUT_DIR, exist_ok=True)

edge_files = ["ap.txt","pa.txt","cp.txt","pc.txt","pt.txt","tp.txt"]

for fn in edge_files:
    path_in  = os.path.join(RAW_DIR,  fn)
    path_out = os.path.join(OUT_DIR, fn)

    # load all edges
    edges = np.loadtxt(path_in, dtype=int)
    if edges.ndim == 1:
        edges = edges[np.newaxis,:]

    # filter: both endpoints ≤ MAX_ID
    mask = (edges[:,0] <= MAX_ID) & (edges[:,1] <= MAX_ID)
    edges = edges[mask]

    # collect all unique node IDs that survive
    uniq = np.unique(edges.flatten())
    # build a mapping old_id → new_id = 0…len(uniq)-1
    new_id = {old:i for i,old in enumerate(uniq)}

    # remap edges
    remapped = np.vectorize(new_id.get)(edges)

    # save
    np.savetxt(path_out, remapped, fmt="%d")
    print(f"{fn}: kept {len(remapped)} edges, remapped {len(uniq)} nodes")
