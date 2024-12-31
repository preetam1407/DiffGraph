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

