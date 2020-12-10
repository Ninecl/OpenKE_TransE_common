import numpy as np
import torch

def get_entity_feature(num_entity, num_relation, triplets):
    """
    构建一个三维数组
    第一维是所有实体数
    第二维表示所有关系数
    第三维表示头实体或尾实体
    这个三维数组表示的是
    这个实体存在几个三元组是某关系的头/尾实体
    """
    entity_feature = np.zeros((num_entity, 2, num_relation))
    for triplet in triplets:
        h_id = triplet[0]
        t_id = triplet[1]
        r_id = triplet[2]
        entity_feature[h_id][0][r_id] += 1
        entity_feature[t_id][1][r_id] += 1
    return torch.IntTensor(entity_feature)


def caculate_feature_weight(entities_feature):
		entities_feature_weight = entities_feature.float()
		for i in range(len(entities_feature_weight)):
			if entities_feature_weight[i].sum() != 0:
				entities_feature_weight[i] = entities_feature_weight[i] / entities_feature_weight[i].sum()
		return entities_feature_weight


def pool_feature(entities_feature_weight, common_head_embeddings, common_tail_embeddings):
    e_f = entities_feature_weight.permute(1, 0, 2)
    e_h_f = e_f[0].view(e_f[0].shape[0], e_f[0].shape[-1], 1)
    e_t_f = e_f[1].view(e_f[1].shape[0], e_f[1].shape[-1], 1)
    # e_f_emb = torch.cat((e_h_f * self.common_head_embeddings.weight, 
    # 				e_t_f * self.common_tail_embeddings.weight), 1).sum(dim=1)
    e_h_f_emb = (e_h_f * common_head_embeddings).sum(dim=1)
    e_t_f_emb = (e_t_f * common_tail_embeddings).sum(dim=1)
    e_f_emb = e_h_f_emb + e_t_f_emb
    return e_f_emb