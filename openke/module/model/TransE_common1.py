import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransE_common1(Model):
	"""
	该TansE_common使用：head_entity + (r_common_tail - r_common_head) = tail_entity
	loss函数修改为 |h + rct - rch - t| + |h - pool_h| + |h - pool_t|
	Args:
		Model ([type]): [description]
	"""
	def __init__(self, ent_tot, rel_tot, entities_feature, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransE_common1, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.entities_feature = entities_feature
		self.entities_feature_weight = self.caculate_feature_weight()

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.common_head_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.common_tail_embeddings = nn.Embedding(self.rel_tot, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.common_head_embeddings.weight.data)
			nn.init.xavier_uniform_(self.common_tail_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.common_head_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.common_tail_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

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
		return entities_feature_weight.cuda()


	def pool_feature(self):
		e_f = self.entities_feature_weight.permute(1, 0, 2)
		e_h_f = e_f[0].view(self.ent_tot, self.rel_tot, 1)
		e_t_f = e_f[1].view(self.ent_tot, self.rel_tot, 1)
		# e_f_emb = torch.cat((e_h_f * self.common_head_embeddings.weight, 
		# 				e_t_f * self.common_tail_embeddings.weight), 1).sum(dim=1)
		e_h_f_emb = (e_h_f * self.common_head_embeddings.weight).sum(dim=1)
		e_t_f_emb = (e_t_f * self.common_tail_embeddings.weight).sum(dim=1)
		e_f_emb = e_h_f_emb + e_t_f_emb
		return e_f_emb


	def _calc(self, h, t, c_r_h, c_r_t, pf_h, pf_t, mode):
		score2 = torch.zeros([0])
		score3 = torch.zeros([0])
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			t = F.normalize(t, 2, -1)
			c_r_h = F.normalize(c_r_h, 2, -1)
			c_r_t = F.normalize(c_r_t, 2, -1)
			pf_h = F.normalize(pf_h, 2, -1)
			pf_t = F.normalize(pf_t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, c_r_h.shape[0], h.shape[-1])
			t = t.view(-1, c_r_t.shape[0], t.shape[-1])
			c_r_h = c_r_h.view(-1, c_r_h.shape[0], c_r_h.shape[-1])
			c_r_t = c_r_t.view(-1, c_r_t.shape[0], c_r_t.shape[-1])

		if mode == 'head_batch':
			score1 = h + (c_r_t - c_r_h - t)
		elif mode == 'tail_batch':
			score1 = (t + c_r_h - c_r_t) - h
		else:
			score1_hrt = h + (c_r_t - c_r_h) - t
			score1_trh = t + (c_r_h - c_r_t) - h
			score1 = torch.cat((score1_hrt, score1_trh), dim=0)
			score2 = h - pf_h
			score3 = t - pf_t

		score1 = torch.norm(score1, self.p_norm, -1).flatten()
		score2 = torch.norm(score2, 2, -1).flatten()
		score3 = torch.norm(score3, 2, -1).flatten()
		return score1, torch.cat((score2, score3), dim=0)


	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		c_r_h = self.common_head_embeddings(batch_r)
		c_r_t = self.common_tail_embeddings(batch_r)
		pf_h = torch.zeros([0])
		pf_t = torch.zeros([0])
		if mode == 'normal':
			entities_feature_embedding = self.pool_feature()
			pf_h = entities_feature_embedding[batch_h]
			pf_t = entities_feature_embedding[batch_t]
		score_trans, score_feature = self._calc(h, t, c_r_h, c_r_t, pf_h, pf_t, mode)
		if self.margin_flag:
			return self.margin - score_trans, score_feature
		else:
			return score_trans, score_feature


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
		score_trans, score_feature = self.forward(data)
		if self.margin_flag:
			score_trans = self.margin - score_trans
			return score_trans.cpu().data.numpy()
		else:
			return score_trans.cpu().data.numpy()