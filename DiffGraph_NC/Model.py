from statistics import mean
import torch
from torch import nn
import torch.nn.functional as F
from params import args
from sklearn.metrics import roc_auc_score
import numpy as np
import math

from Utils.Utils import cal_infonce_loss
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class HGDM(nn.Module):
    def __init__(self,f_dim):
        super(HGDM, self).__init__()
        out_dims = eval(args.dims) + [args.latdim]
        in_dims = out_dims[::-1]
        self.user_denoise_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm)
        self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps)
       
        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.helayers1 = nn.ModuleList()
        self.helayers2 = nn.ModuleList()
        self.main_layers = nn.ModuleList()
        self.weight = False
        for i in range(0, args.gcn_layer):
            self.helayers1.append(UUGCNLayer(args.latdim, args.latdim, weight=self.weight, bias=False, activation=self.act))
        for i in range(0, args.gcn_layer):
            self.helayers2.append(UUGCNLayer(args.latdim, args.latdim, weight=self.weight, bias=False, activation=self.act))
        for i in range(0, args.uugcn_layer):
            self.main_layers.append(UUGCNLayer(args.latdim, args.latdim, weight=self.weight, bias=False, activation=self.act))
        
        self.transform_layer = torch.nn.Linear(f_dim,args.latdim,bias=True)
        nn.init.xavier_normal_(self.transform_layer.weight, gain=1.414)
        self.dense = torch.nn.Linear(args.latdim,4)
        
    
        self.pool = 'sum'
    def forward(self, he_adjs,feature_list, is_training=True):
        
        
        embed = self.transform_layer(feature_list)
        target_embedding = [embed]
        source_embeddings1 = [embed]
        source_embeddings2= [embed]
      
        for i, layer in enumerate(self.main_layers):
            embeddings = layer(he_adjs[0], target_embedding[-1])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            target_embedding += [norm_embeddings]

        target_embedding = sum(target_embedding)

        for i, layer in enumerate(self.helayers1):
            embeddings = layer(he_adjs[1], source_embeddings1[-1])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            source_embeddings1 += [norm_embeddings]

        source_embeddings1 = sum(source_embeddings1)

        for i, layer in enumerate(self.helayers2):
            embeddings = layer(he_adjs[2], source_embeddings2[-1])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            source_embeddings2 += [norm_embeddings]

        source_embeddings2 = sum(source_embeddings2)

        return source_embeddings1,source_embeddings2,target_embedding
    
    def cal_loss(self, ancs, label,he_adjs,initial_feature):
        source_embeddings1,source_embeddings2,target_embedding = self.forward(he_adjs,initial_feature)
   
        source_embeddings = source_embeddings1+source_embeddings2

        diff_loss,diff_embeddings= self.diffusion_model.training_losses2(self.user_denoise_model, target_embedding, source_embeddings, ancs)
        diff_loss = diff_loss.mean()
        all_embeddings = target_embedding+diff_embeddings
        scores = self.dense(all_embeddings)
        scores = F.log_softmax(scores,dim=1)

        batch_u = scores[ancs]
        batch_label = torch.argmax(label[ancs], dim=-1)
        nll_loss = F.nll_loss(batch_u,batch_label)
        return nll_loss,diff_loss
    def get_embeds(self, ancs, label,he_adjs,initial_feature):
        source_embeddings1,source_embeddings2,target_embedding = self.forward(he_adjs,initial_feature)
   
        source_embeddings = source_embeddings1+source_embeddings2

        diff_embeddings= self.diffusion_model.p_sample(self.user_denoise_model, source_embeddings, args.sampling_steps)
        
        all_embeddings = target_embedding+diff_embeddings
        
        return all_embeddings[ancs]
    def get_allembeds(self, he_adjs,initial_feature):
        source_embeddings1,source_embeddings2,target_embedding = self.forward(he_adjs,initial_feature)
   
        source_embeddings = source_embeddings1+source_embeddings2

        diff_embeddings= self.diffusion_model.p_sample(self.user_denoise_model, source_embeddings, args.sampling_steps)
        all_embeddings = target_embedding+diff_embeddings
        scores = self.dense(all_embeddings)
        return all_embeddings,scores

class DGLLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(DGLLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            # self.e_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
            xavier_uniform_(self.v_w)
            # init.xavier_uniform_(self.e_w)
        self._activation = activation

    # def forward(self, graph, feat):
    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
                v_f = torch.mm(v_f, self.v_w)
                # e_f = t.mm(e_f, self.e_w)
            node_f = torch.cat([u_f, v_f], dim=0)
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            # graph.edata['e_f'] = e_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

class UUGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(UUGCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            init(self.u_w)
        self._activation = activation

    # def forward(self, graph, feat):
    def forward(self, graph, u_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
            node_f = u_f
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            # graph.edata['e_f'] = e_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

        
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(device)
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)
	
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]]).to(device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(device)]).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).to(device)
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            x_t = model_mean
        return x_t
            
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
        return model_mean, model_log_variance

    def training_losses(self, model, targetEmbeds, x_start):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(targetEmbeds, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)
        # mse = cal_infonce_loss(targetEmbeds,model_output,args.temp)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse
        # cal_loss = cal_infonce_loss(model_output,targetEmbeds,args.temp)
        # return diff_loss, cal_loss,model_output
        return diff_loss,model_output
    
    def training_losses2(self, model, targetEmbeds, x_start, batch):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        diff_loss = weight * mse
        diff_loss = diff_loss[batch]

        # mse = cal_infonce_loss(targetEmbeds[batch],model_output[batch],args.temp)
        # weight = self.SNR(ts - 1) - self.SNR(ts)
        # weight = torch.where((ts == 0), 1.0, weight)
        # diff_loss = weight[batch]*mse


        # cal_loss = cal_infonce_loss(model_output,targetEmbeds,args.temp)
        # return diff_loss, cal_loss,model_output
        return diff_loss,model_output
		
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    

class MAE(nn.Module):
    def __init__(self, in_dims, out_dims, norm=False, dropout=0.5):
        super(MAE, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.norm = norm

        in_dims_temp = self.in_dims
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        #
        # size = self.emb_layer.weight.size()
        # fan_out = size[0]
        # fan_in = size[1]
        # std = np.sqrt(2.0 / (fan_in + fan_out))
        # self.emb_layer.weight.data.normal_(0.0, std)
        # self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, targetEmbeds,x, ancs,is_training=True):
        
        x_start = x
        if self.norm:
            x = F.normalize(x)
        keepsample = (torch.rand(x.shape) < 0.8) * 1.0
        keepsample = keepsample.to(device)
        h = x * keepsample

        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        diff_loss = self.training_losses(targetEmbeds, h,ancs)
        if is_training:
            return diff_loss,h
        return h

    def training_losses(self, targetEmbeds, model_output,batch):
        
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)
        diff_loss = mse[batch]
        return diff_loss
    def mean_flat(self,tensor):
   
        return tensor.mean(dim=list(range(1, len(tensor.shape))))


class MetaPathLearner(nn.Module):
    """
    GTN-style learnable meta-path block.
    Input: A_orig (E x N x N) tensor of single-hop adjacencies.
    Output: A_learned (C x N x N) tensor of learned 2-hop meta-path adjacencies.
    """
    def __init__(self, num_edge_types, num_channels):
        super(MetaPathLearner, self).__init__()
        self.conv1 = nn.Linear(num_edge_types, num_channels, bias=False)
        self.conv2 = nn.Linear(num_edge_types, num_channels, bias=False)
        self.conv3 = nn.Linear(num_edge_types, num_channels, bias=False)

    def forward(self, A_orig):
        # A_orig: (E, N, N)
        # Soft selections α1, α2: (C, E)
        α1 = F.softmax(self.conv1.weight, dim=1)
        α2 = F.softmax(self.conv2.weight, dim=1)
        α3 = F.softmax(self.conv3.weight, dim=1)  # (C, E)
        print("α3 channel 0:", α3[0])
        # Build Q1, Q2: (C, N, N)
        Q1 = torch.einsum('ce,enk->cnk', α1, A_orig)
        Q2 = torch.einsum('ce,enk->cnk', α2, A_orig)
        Q3 = torch.einsum('ce,enk->cnk', α3, A_orig)
        # Compose 2-hop meta-paths per channel
        A_learned = torch.stack([Q1[c] @ Q2[c] @ Q3[c]
                                 for c in range(α1.size(0))], dim=0)
        return A_learned
    


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_sparse
# from torch_geometric.utils import softmax
# from gcn import GCNConv
# from Utils.Utils import _norm, generate_non_local_graph

# class FastGTNs(nn.Module):
#     def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None):
#         super().__init__()
#         self.args = args
#         self.num_nodes = num_nodes
#         self.num_FastGTN_layers = args.num_FastGTN_layers
#         self.fastGTNs = nn.ModuleList([
#             FastGTN(
#                 num_edge_type if i==0 else args.node_dim,
#                 w_in if i==0 else args.node_dim,
#                 num_class,
#                 num_nodes,
#                 args
#             )
#             for i in range(self.num_FastGTN_layers)
#         ])
#         self.linear = nn.Linear(args.node_dim, num_class)
#         self.loss = nn.BCELoss() if args.dataset=='PPI' else nn.CrossEntropyLoss()

#     def forward(self, A_list, X, target_x, target=None, eval=False, **kwargs):
#         H_, Ws = self.fastGTNs[0](A_list, X, self.num_nodes, **kwargs)
#         for i in range(1, self.num_FastGTN_layers):
#             H_, Ws = self.fastGTNs[i](A_list, H_, self.num_nodes, **kwargs)
#         y = self.linear(H_[target_x])
#         if eval:
#             return y
#         return (self.loss(torch.sigmoid(y) if self.args.dataset=='PPI' else y, target), y, Ws)

# class FastGTN(nn.Module):
#     def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None):
#         super().__init__()
#         self.args = args
#         self.num_edge_type = num_edge_type + (1 if args.non_local else 0)
#         self.num_channels = args.num_channels
#         self.num_nodes = num_nodes
#         self.w_in = w_in
#         self.w_out = args.node_dim
#         self.num_layers = args.num_layers

#         self.layers = nn.ModuleList([
#             FastGTLayer(
#                 self.num_edge_type,
#                 self.num_channels,
#                 num_nodes,
#                 first=(i==0),
#                 args=args
#             )
#             for i in range(self.num_layers)
#         ])

#         # initial channel projections
#         self.Ws = nn.ParameterList([
#             GCNConv(in_channels=w_in, out_channels=self.w_out).weight
#             for _ in range(self.num_channels)
#         ])
#         self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
#         self.feat_trans_layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(self.w_out,128), nn.ReLU(), nn.Linear(128,64))
#             for _ in range(self.num_layers+1)
#         ])
#         self.relu = nn.ReLU()
#         self.out_norm = nn.LayerNorm(self.w_out)

#     def forward(self, A, X, num_nodes, eval=False, epoch=None, layer=None):
#         Ws, H = [], []
#         # project X into each channel
#         X_ = [X @ W for W in self.Ws]
#         H  = [X @ W for W in self.Ws]

#         for i, gtlayer in enumerate(self.layers):
#             # optional non-local graph update
#             if self.args.non_local:
#                 g_new = generate_non_local_graph(
#                     self.args, self.feat_trans_layers[i], torch.stack(H).mean(dim=0),
#                     A, self.num_edge_type, num_nodes
#                 )
#                 deg_inv_sqrt, deg_row, deg_col = _norm(g_new[0].detach(), num_nodes, g_new[1])
#                 g_new[1] = softmax(g_new[1], deg_row)
#                 if len(A) < self.num_edge_type:
#                     A.append(g_new)
#                 else:
#                     A[-1] = g_new

#             H, W = gtlayer(H, A, num_nodes, epoch=epoch, layer=i+1)
#             Ws.append(W)

#         # blend and aggregate channels
#         for i in range(self.num_channels):
#             comb = self.args.beta * X_[i] + (1-self.args.beta) * H[i]
#             if i==0:
#                 H_ = F.relu(comb)
#             else:
#                 if self.args.channel_agg=='concat':
#                     H_ = torch.cat((H_, F.relu(comb)), dim=1)
#                 else:
#                     H_ = H_ + F.relu(comb)

#         if self.args.channel_agg=='concat':
#             H_ = F.relu(self.linear1(H_))
#         else:
#             H_ = H_ / self.num_channels

#         return H_, Ws

# class FastGTLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, num_nodes, first=True, args=None):
#         super().__init__()
#         self.conv1 = FastGTConv(in_channels, out_channels, num_nodes, args=args)
#         self.first = first
#         self.num_nodes = num_nodes

#     def forward(self, H_, A, num_nodes, epoch=None, layer=None):
#         result_A, W1 = self.conv1(A, num_nodes, epoch=epoch, layer=layer)
#         Hs = []
#         for i, (edge_index, edge_value) in enumerate(result_A):
#             mat = torch.sparse_coo_tensor(
#                 edge_index, edge_value, (num_nodes, num_nodes)
#             ).to(edge_index.device)
#             Hs.append(torch.sparse.mm(mat, H_[i]))
#         return Hs, W1

# class FastGTConv(nn.Module):
#     def __init__(self, in_channels, out_channels, num_nodes, args=None):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
#         self.bias = None
#         self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
#         self.args = args
#         self.num_nodes = num_nodes
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.normal_(self.weight, std=0.1)
#         if self.args.non_local and self.args.non_local_weight != 0:
#             with torch.no_grad():
#                 self.weight[:, -1] = self.args.non_local_weight

#     def forward(self, A, num_nodes, epoch=None, layer=None):
#         filter = F.softmax(self.weight, dim=1)
#         results = []
#         for i in range(filter.size(0)):
#             total_idx = None
#             total_val = None
#             for j, (edge_index, edge_value) in enumerate(A):
#                 w = filter[i, j]
#                 if total_idx is None:
#                     total_idx = edge_index
#                     total_val = edge_value * w
#                 else:
#                     total_idx = torch.cat((total_idx, edge_index), dim=1)
#                     total_val = torch.cat((total_val, edge_value * w), dim=0)
#             idx, val = torch_sparse.coalesce(
#                 total_idx, total_val, m=num_nodes, n=num_nodes, op='add'
#             )
#             results.append((idx, val))
#         return results, filter
#     import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.init import xavier_normal_

# class FastGTNWithDiffusion(nn.Module):
#     """
#     Runs FastGTN backbone over the full heterogeneous graph,
#     pads author-only features to full node set,
#     slices out author embeddings,
#     then applies diffusion + classification from HGDM.
#     """
#     def __init__(self,
#                  f_dim,
#                  num_edge_type,
#                  w_in,
#                  num_class,
#                  num_nodes,
#                  args):
#         super(FastGTNWithDiffusion, self).__init__()
#         self.args = args
#         self.num_nodes = num_nodes
#         # 1) FastGTN backbone
#         self.fastgtn = FastGTNs(num_edge_type,
#                                 w_in,
#                                 num_class,
#                                 num_nodes,
#                                 args)
#         # 2) transform initial author features
#         self.transform_layer = nn.Linear(f_dim, args.latdim, bias=True)
#         xavier_normal_(self.transform_layer.weight, gain=1.414)
#         # 3) diffusion modules
#         out_dims = eval(args.dims) + [args.latdim]
#         in_dims  = out_dims[::-1]
#         self.user_denoise_model = Denoise(in_dims,
#                                           out_dims,
#                                           args.d_emb_size,
#                                           norm=args.norm)
#         self.diffusion_model    = GaussianDiffusion(args.noise_scale,
#                                                    args.noise_min,
#                                                    args.noise_max,
#                                                    args.steps)
#         # 4) final classifier
#         self.dense = nn.Linear(args.latdim, num_class)

#     def cal_loss(self, ancs, label, he_adjs, feature_list):
#         # -- 1) initial author embeddings
#         init_embed = self.transform_layer(feature_list)    # (NA, latdim)
#         NA = init_embed.size(0)
#         # -- 2) pad to full graph features
#         device = init_embed.device
#         X_full = torch.zeros((self.num_nodes, init_embed.size(1)), device=device)
#         X_full[:NA] = init_embed
#         # -- 3) forward through FastGTN backbone
#         H_full, _ = self.fastgtn.fastGTNs[0](he_adjs, X_full, self.num_nodes)
#         for i in range(1, self.fastgtn.num_FastGTN_layers):
#             H_full, _ = self.fastgtn.fastGTNs[i](he_adjs, H_full, self.num_nodes)
#         # H_full: (N_tot, latdim)
#         # -- 4) slice out authors
#         author_embed = H_full[:NA]                        # (NA, latdim)
#         # -- 5) diffusion
#         diff_loss, diff_embeds = self.diffusion_model.training_losses2(
#             self.user_denoise_model,
#             author_embed,
#             init_embed,
#             ancs)
#         diff_loss = diff_loss.mean()
#         # -- 6) classification
#         all_embeds = author_embed + diff_embeds
#         logits     = self.dense(all_embeds)
#         logp       = F.log_softmax(logits, dim=1)
#         batch_logits = logp[ancs]
#         batch_labels = torch.argmax(label[ancs], dim=-1)
#         nll_loss = F.nll_loss(batch_logits, batch_labels)
#         return nll_loss, diff_loss

#     def get_embeds(self, ancs, label, he_adjs, feature_list):
#         init_embed = self.transform_layer(feature_list)
#         NA = init_embed.size(0)
#         X_full = torch.zeros((self.num_nodes, init_embed.size(1)), device=init_embed.device)
#         X_full[:NA] = init_embed
#         H_full, _ = self.fastgtn.fastGTNs[0](he_adjs, X_full, self.num_nodes)
#         for i in range(1, self.fastgtn.num_FastGTN_layers):
#             H_full, _ = self.fastgtn.fastGTNs[i](he_adjs, H_full, self.num_nodes)
#         author_embed = H_full[:NA]
#         diff_embeds = self.diffusion_model.p_sample(
#             self.user_denoise_model,
#             author_embed,
#             self.args.sampling_steps)
#         all_embeds = author_embed + diff_embeds
#         return all_embeds[ancs]

#     def get_allembeds(self, he_adjs, feature_list):
#         init_embed = self.transform_layer(feature_list)
#         NA = init_embed.size(0)
#         X_full = torch.zeros((self.num_nodes, init_embed.size(1)), device=init_embed.device)
#         X_full[:NA] = init_embed
#         H_full, _ = self.fastgtn.fastGTNs[0](he_adjs, X_full, self.num_nodes)
#         for i in range(1, self.fastgtn.num_FastGTN_layers):
#             H_full, _ = self.fastgtn.fastGTNs[i](he_adjs, H_full, self.num_nodes)
#         author_embed = H_full[:NA]
#         diff_embeds = self.diffusion_model.p_sample(
#             self.user_denoise_model,
#             author_embed,
#             self.args.sampling_steps)
#         all_embeds = author_embed + diff_embeds
#         scores = self.dense(all_embeds)
#         return all_embeds, scores
