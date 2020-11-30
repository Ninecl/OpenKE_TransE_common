import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransE_with_feature(Model):

	def __init__(self, ent_tot, rel_tot, entities_feature, common_head_embeddings, common_tail_embeddings, 
              dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransE_with_feature, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.common_head_embeddings = common_head_embeddings
		self.common_tail_embeddings = common_tail_embeddings
		self.entities_feature = entities_feature
		self.entities_feature_weight = self.caculate_feature_weight()

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		self.ent_embeddings.weight.data = self.pool_feature()
		self.rel_embeddings.weight.data = self.common_tail_embeddings - self.common_head_embeddings

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def caculate_feature_weight(self):
		entities_feature_weight = self.entities_feature.float()
		for i in range(len(entities_feature_weight)):
			if entities_feature_weight[i].sum() != 0:
				entities_feature_weight[i] = entities_feature_weight[i] / entities_feature_weight[i].sum()
		return entities_feature_weight


	def pool_feature(self):
		e_f = self.entities_feature_weight.permute(1, 0, 2).cuda()
		e_h_f = e_f[0].view(e_f[0].shape[0], self.rel_tot, 1)
		e_t_f = e_f[1].view(e_f[1].shape[0], self.rel_tot, 1)
		# e_f_emb = torch.cat((e_h_f * self.common_head_embeddings.weight, 
		# 				e_t_f * self.common_tail_embeddings.weight), 1).sum(dim=1)
		e_h_f_emb = (e_h_f * self.common_head_embeddings).sum(dim=1)
		e_t_f_emb = (e_t_f * self.common_tail_embeddings).sum(dim=1)
		e_f_emb = e_h_f_emb + e_t_f_emb
		return e_f_emb


	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()