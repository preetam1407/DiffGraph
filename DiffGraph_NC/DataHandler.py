
import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import dgl
from params import args
from Utils.TimeLogger import log
import torch as t
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from Model import MetaPathLearner

device = "cuda:0" if t.cuda.is_available() else "cpu"

class DataHandler:
    def __init__(self):
        if args.data == 'DBLP':
            self.predir = './data/DBLP/'
        elif args.data == 'aminer':
            self.predir = './data/aminer/'
        else:
            self.predir = './data/Freebase/'

        self.global_mats = None
        self.NA = None
        self.Ntot = None
        self.device = device
        self.metapath_learner: MetaPathLearner = None


    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) is not coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        # symmetric normalize a scipy coo_matrix
        degree = np.array(mat.sum(axis=-1)).flatten()
        dInvSqrt = np.power(degree, -0.5)
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        D = sp.diags(dInvSqrt)
        return (D @ mat @ D).tocoo()

    def makeTorchAdj(self, mat):
        # unchanged bipartite UI adjacency builder
        user, item = mat.shape
        a = sp.csr_matrix((user, user))
        b = sp.csr_matrix((item, item))
        big = sp.vstack([
            sp.hstack([a, mat]),
            sp.hstack([mat.transpose(), b])
        ]).tocoo()
        big.data[:] = 1.0
        big = self.normalizeAdj(big)
        idxs = t.from_numpy(np.vstack([big.row, big.col]).astype(np.int64))
        vals  = t.from_numpy(big.data.astype(np.float32))
        return t.sparse.FloatTensor(idxs, vals, big.shape).to(device)

    def makeTorchuAdj(self, mat):
        # unchanged single-type adjacency builder
        A = (mat != 0).astype(float) + sp.eye(mat.shape[0])
        A = self.normalizeAdj(A.tocoo())
        idxs = t.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
        vals  = t.from_numpy(A.data.astype(np.float32))
        return t.sparse.FloatTensor(idxs, vals, A.shape).to(device)

    def makeBiAdj(self, mat):
        # unchanged bipartite UI DGL graph
        n_user, n_item = mat.shape
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        big = sp.vstack([
            sp.hstack([a, mat]),
            sp.hstack([mat.transpose(), b])
        ]).tocoo()
        src, dst = big.row, big.col
        return dgl.graph((src, dst), idtype=t.int32, num_nodes=big.shape[0]).to(device)
    

    def LoadData(self):
        if args.data == 'DBLP':
        # 1) load single‐hop mats + features + splits
            features, raw_mats, train, val, test, labels = self.load_dblp_data()
            self.feature_list = t.FloatTensor(features).to(device)

            # 2) build global offsets
            NA, NP = raw_mats[0].shape     # A→P
            NC     = raw_mats[2].shape[0]  # C→P
            NT     = raw_mats[4].shape[1]  # P→T
            off_A, off_P, off_C, off_T = 0, NA, NA+NP, NA+NP+NC
            N_tot = NA + NP + NC + NT

            def make_global(mat, src, dst):
                rows = mat.row + src
                cols = mat.col + dst
                return sp.coo_matrix((mat.data, (rows, cols)), shape=(N_tot, N_tot))

            # assemble & normalize each of the E=6 raw relations
            relations = [
                (raw_mats[0], off_A, off_P),
                (raw_mats[1], off_P, off_A),
                (raw_mats[2], off_C, off_P),
                (raw_mats[3], off_P, off_C),
                (raw_mats[4], off_P, off_T),
                (raw_mats[5], off_T, off_P),
            ]

            global_mats = []
            for mat, s, d in relations:
                G = make_global(mat, s, d).astype(np.float32)
                G.data[:] = 1.0
                # G = self.normalizeAdj(G).tocoo()
                global_mats.append(G)

            for i, G in enumerate(global_mats):
                print(f"  global_mats[{i}]: nnz={G.nnz}, shape={G.shape}")

            C = args.num_channels
            E = len(global_mats)

            # ─── Option A:Dense GTN (just for debugging) ───
            # # """
            # # stack into one dense tensor [E × N_tot × N_tot]
            # A_orig = t.stack([
            #     t.from_numpy(global_mats[e].toarray()).float()
            #     for e in range(E)
            # ], dim=0).to(device)

            # # run your MetaPathLearner → produces [C × N_tot × N_tot]
            # A_learned = self.metapath_learner(A_orig)

            # # slice out author–author
            # A_auth = A_learned[:, :NA, :NA]   # (C, NA, NA)

            # # to DGL
            # self.he_adjs = []
            # for c in range(C):
            #     mat_sp = sp.coo_matrix(A_auth[c].detach().cpu().numpy())
            #     mat_sp = self.normalizeAdj(mat_sp)
            #     self.he_adjs.append(dgl.from_scipy(mat_sp).to(device))
            # # """

            # ─── Option B: Memory‐efficient Sparse GTN ───
            α1 = F.softmax(self.metapath_learner.conv1.weight, dim=1).detach().cpu().numpy()  
            α2 = F.softmax(self.metapath_learner.conv2.weight, dim=1).detach().cpu().numpy()
            
            self.he_adjs = []
            for c in range(C):
                # build sparse Q₁, Q₂ (and Q₃ if you added it)
                Q1 = sum(global_mats[e] * α1[c,e] for e in range(E))
                Q2 = sum(global_mats[e] * α2[c,e] for e in range(E))
                
                # 2‐hop composition in sparse form
                Hc = Q1.dot(Q2)        
              
                Hc = (Hc != 0).astype(np.float32)

                # slice out author–author block
                Hc_csr = Hc.tocsr()
                Haa    = Hc_csr[:NA, :NA]  

                # re‐binarize & normalize this small AA block
                Haa.data[:] = 1.0
                Haa = self.normalizeAdj(Haa)  
                print(f"Channel {c}: {Haa.nnz} non-zero entries in the author–author block")
             
                print(f"  unique values in data: {np.unique(Haa.data)}")

                # to DGL
                self.he_adjs.append(dgl.from_scipy(Haa).to(device))

            # finally store your splits & labels
            self.train_idx, self.val_idx, self.test_idx = train, val, test
            self.labels = labels

            
        if args.data == 'Freebase':
            # unchanged...
            features_list, mam_mat, mdm_mat, mwm_mat, train, val, test, labels = self.load_Freebase_data()
            self.feature_list = t.FloatTensor(features_list).to(device)
            self.hete_adj1 = dgl.from_scipy(mam_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(mdm_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(mwm_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels
            self.he_adjs = [self.hete_adj1, self.hete_adj2, self.hete_adj3]

        if args.data == 'aminer':
            # unchanged...
            features_list, pap_mat, prp_mat, pos_mat, train, val, test, labels = self.load_aminer_data()
            self.feature_list = t.FloatTensor(features_list).to(device)
            self.hete_adj1 = dgl.from_scipy(pap_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(prp_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(pos_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels
            self.he_adjs = [self.hete_adj1, self.hete_adj2, self.hete_adj3]

    
    def load_dblp_data(self):
        import os, numpy as np, scipy.sparse as sp, torch as t
        from sklearn.preprocessing import OneHotEncoder

        # 1) author features → NA
        features_a = sp.load_npz(self.predir + 'a_feat.npz').astype('float32')
        NA = features_a.shape[0]
        features_a = t.FloatTensor(preprocess_features(features_a))

        # 2) read edge‐lists
        edge_files = ['ap.txt','pa.txt','cp.txt','pc.txt','pt.txt','tp.txt']
        edge_arrays = []
        for fn in edge_files:
            arr = np.loadtxt(os.path.join(self.predir, fn), dtype=int)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            edge_arrays.append(arr)
        ap, pa, cp, pc, pt, tp = edge_arrays

        # 3) infer node-type counts
        NP = max(ap[:,1].max(), pa[:,0].max()) + 1
        NC = max(cp[:,0].max(), pc[:,1].max()) + 1
        NT = max(pt[:,1].max(), tp[:,0].max()) + 1

        # 4) build raw adjacency matrices (COO)
        raw_mats = [
            sp.coo_matrix((np.ones(len(ap)), (ap[:,0], ap[:,1])), shape=(NA, NP)),
            sp.coo_matrix((np.ones(len(pa)), (pa[:,0], pa[:,1])), shape=(NP, NA)),
            sp.coo_matrix((np.ones(len(cp)), (cp[:,0], cp[:,1])), shape=(NC, NP)),
            sp.coo_matrix((np.ones(len(pc)), (pc[:,0], pc[:,1])), shape=(NP, NC)),
            sp.coo_matrix((np.ones(len(pt)), (pt[:,0], pt[:,1])), shape=(NP, NT)),
            sp.coo_matrix((np.ones(len(tp)), (tp[:,0], tp[:,1])), shape=(NT, NP)),
        ]

        # 5) load labels
        labels = np.load(self.predir + 'labels.npy')
        enc = OneHotEncoder(sparse=False).fit(labels.reshape(-1,1))
        labels = t.FloatTensor(enc.transform(labels.reshape(-1,1))).to(self.device)

        # 6) load splits
        train = [t.LongTensor(np.load(self.predir + f"train_{i}.npy")) for i in args.ratio]
        val   = [t.LongTensor(np.load(self.predir + f"val_{i}.npy"))   for i in args.ratio]
        test  = [t.LongTensor(np.load(self.predir + f"test_{i}.npy"))  for i in args.ratio]

        return features_a, raw_mats, train, val, test, labels

def preprocess_features(features):
    rowsum = np.array(features.sum(1)).flatten()
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features).todense()

def encode_onehot(labels):
    labels = labels.reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(labels)
    return enc.transform(labels).toarray()

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)
    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[
            (self.iter_counter - 1) * self.batch_size:
             self.iter_counter * self.batch_size
        ])
    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))
    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter
    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
