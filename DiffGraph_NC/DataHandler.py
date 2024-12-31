import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
import dgl
from Utils.TimeLogger import log
import torch as t
from sklearn.preprocessing import OneHotEncoder
device = "cuda:0" if t.cuda.is_available() else "cpu"

class DataHandler:
    def __init__(self):

        if args.data == 'DBLP':
            predir = './data/DBLP/'
        if args.data == 'aminer':
            predir = './data/aminer/'
        self.predir = predir


    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        user,item = mat.shape[0],mat.shape[1]
        a = sp.csr_matrix((user, user))
        b = sp.csr_matrix((item, item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)

    def makeTorchuAdj(self, mat):
        """Create tensor-based adjacency matrix for user social graph.

        Args:
            mat: Adjacency matrix.

        Returns:
            Tensor-based adjacency matrix.
        """
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)
    def makeBiAdj(self, mat):
        n_user = mat.shape[0]
        n_item = mat.shape[1]
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=t.int32,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph


    def LoadData(self):
        if args.data == 'DBLP':
            features_list,apa_mat,ata_mat,ava_mat,train,val,test,labels = self.load_dblp_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.hete_adj1 = dgl.from_scipy(apa_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(ata_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(ava_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels
        
        

            # self.train_idx_generator = index_generator(batch_size=args.batch, indices=self.train_idx)
            # self.val_idx_generator = index_generator(batch_size=args.batch, indices=self.val_idx, shuffle=False)
            # self.test_idx_generator = index_generator(batch_size=args.batch, indices=self.test_idx, shuffle=False)
            self.he_adjs = [self.hete_adj1,self.hete_adj2,self.hete_adj3]
        if args.data == 'Freebase':
            features_list,mam_mat,mdm_mat,mwm_mat,train,val,test,labels = self.load_Freebase_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.hete_adj1 = dgl.from_scipy(mam_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(mdm_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(mwm_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels

            self.he_adjs = [self.hete_adj1,self.hete_adj2,self.hete_adj3]

        if args.data == 'aminer':
            features_list,pap_mat,prp_mat,pos_mat,train,val,test,labels = self.load_aminer_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.hete_adj1 = dgl.from_scipy(pap_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(prp_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(pos_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels

            self.he_adjs = [self.hete_adj1,self.hete_adj2,self.hete_adj3]

           
          
        
    def load_dblp_data(self):
        features_a = sp.load_npz(self.predir + 'a_feat.npz').astype("float32")
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_a = t.FloatTensor(preprocess_features(features_a))
        
        apa_mat=sp.load_npz(self.predir + "apa.npz")
        ata_mat=sp.load_npz(self.predir + "apcpa.npz")
        ava_mat=sp.load_npz(self.predir + "aptpa.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels= t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_a,apa_mat,ata_mat,ava_mat,train,val,test,labels
    
    def load_Freebase_data(self):
        type_num = [3492, 2502, 33401, 4459]
       
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_m = sp.eye(type_num[0])
        features_m=t.FloatTensor(preprocess_features(features_m))
        mam = sp.load_npz(self.predir + "mam.npz")
        mdm = sp.load_npz(self.predir + "mdm.npz")
        mwm = sp.load_npz(self.predir + "mwm.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels= t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_m,mam,mdm,mwm,train,val,test,labels
    
    def load_aminer_data(self):
        type_num = [6564, 13329, 35890]
       
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_p = sp.eye(type_num[0])
        features_p=t.FloatTensor(preprocess_features(features_p))
        pap = sp.load_npz(self.predir + "pap.npz")
        prp = sp.load_npz(self.predir + "prp.npz")
        pos = sp.load_npz(self.predir + "pos.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels= t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_p,pap,prp,pos,train,val,test,labels



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

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
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


