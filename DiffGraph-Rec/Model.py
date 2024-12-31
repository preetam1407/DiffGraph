from statistics import mean
import torch
from torch import nn
import torch.nn.functional as F
from params import args
import numpy as np
import math
from Utils.Utils import *
import dgl.function as fn
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
from torch.nn.init import xavier_uniform_
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
#Models

class HGDM(nn.Module):
    def __init__(self,data_handler):
        super(HGDM, self).__init__()
        self.n_user = data_handler.userNum
        self.n_item = data_handler.itemNum
        self.behavior_mats = data_handler.behavior_mats
        self.target_adj = data_handler.target_adj
        self.n_hid = args.latdim
        self.n_layers = args.gcn_layer
        self.embedding_dict = self.init_weight(self.n_user, self.n_item, self.n_hid)
        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.layers = nn.ModuleList()
        
        self.hter_layers = nn.ModuleList()
        self.weight = False
        for i in range(0, self.n_layers):
            self.layers.append(DGLLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))
        for i in range(0,len(self.behavior_mats)):
            single_layers = nn.ModuleList()
            for i in range(0, self.n_layers):
                single_layers.append(DGLLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))
            self.hter_layers.append(single_layers)
        self.diffusion_process = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).to(device)
        out_dims = eval(args.dims) + [args.latdim]
        in_dims = out_dims[::-1]
        self.usr_denoiser = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
        self.item_denoiser = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
        self.final_act = nn.LeakyReLU(negative_slope=0.5)

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    def forward(self):

        init_embedding = torch.concat([self.embedding_dict['user_emb'],self.embedding_dict['item_emb']],axis=0)
        init_heter_embedding = torch.concat([self.embedding_dict['user_emb'],self.embedding_dict['item_emb']],axis=0)
        all_embeddings = [init_embedding]
        all_heter_embeddings = []
        
       
        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(self.target_adj, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
            else:
                embeddings = layer(self.target_adj, embeddings[:self.n_user], embeddings[self.n_user:])

            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings += [norm_embeddings]
        ui_embeddings = sum(all_embeddings)


        for i in range(0,len(self.behavior_mats)):
            sub_heter_embeddings = [init_heter_embedding]
            for j, layer in enumerate(self.layers):
                if j == 0:
                    embeddings = layer(self.behavior_mats[i] , self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
                else:
                    embeddings = layer(self.behavior_mats[i] , embeddings[:self.n_user], embeddings[self.n_user:])

                norm_embeddings = F.normalize(embeddings, p=2, dim=1)
                
                sub_heter_embeddings += [norm_embeddings]
            sub_heter_embeddings = sum(sub_heter_embeddings)
            all_heter_embeddings.append(sub_heter_embeddings)
       
        all_heter_embeddings = sum(all_heter_embeddings)
        
        target_user_embedding,target_item_embedding = ui_embeddings[:self.n_user],ui_embeddings[self.n_user:]
        heter_user_embedding,heter_item_embedding = all_heter_embeddings[:self.n_user],all_heter_embeddings[self.n_user:]

        return target_user_embedding,target_item_embedding,heter_user_embedding,heter_item_embedding
    
    def cal_loss(self,ancs,poss,negs):
        usrEmbeds, itmEmbeds, h_usrEmbeds, h_itemEmbeds = self.forward()
        u_diff_loss,diff_usrEmbeds= self.diffusion_process.training_losses2(self.usr_denoiser, usrEmbeds, h_usrEmbeds, ancs)
        i_diff_loss,diff_itemEmbeds = self.diffusion_process.training_losses2(self.item_denoiser, itmEmbeds, h_itemEmbeds, poss)
        diff_loss = (u_diff_loss.mean()+i_diff_loss.mean())
        usrEmbeds = usrEmbeds+diff_usrEmbeds
        itmEmbeds = itmEmbeds+diff_itemEmbeds

        ancEmbeds = usrEmbeds[ancs]
        posEmbeds = itmEmbeds[poss]
        negEmbeds = itmEmbeds[negs]
        
            
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch 
        regLoss = ((torch.norm(ancEmbeds) ** 2 + torch.norm(posEmbeds) ** 2 + torch.norm(negEmbeds) ** 2) * args.reg)/args.batch
        loss = bprLoss + regLoss + diff_loss
        return loss,bprLoss,regLoss,diff_loss
    def predict(self):
        usrEmbeds, itmEmbeds, h_usrEmbeds, h_itemEmbeds = self.forward()
        denoised_u = self.diffusion_process.p_sample(self.usr_denoiser, h_usrEmbeds, args.sampling_steps)
        denoised_i = self.diffusion_process.p_sample(self.item_denoiser, h_itemEmbeds, args.sampling_steps)
        usrEmbeds = usrEmbeds+denoised_u
        itmEmbeds = itmEmbeds+denoised_i
        return usrEmbeds,itmEmbeds

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


        self.history_num_per_term = 10
        self.Lt_history = torch.zeros(steps, 10, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

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
        device = x_start.device
        # ts, pt = self.sample_timesteps(batch_size, device,'importance')
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)
        # mse = cal_infonce_loss(targetEmbeds,model_output,args.temp)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        diff_loss = weight * mse
        diff_loss = diff_loss[batch]
        return diff_loss,model_output
		
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            

            return t, pt

        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError