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
    print("Caculating the feature of entities...")
    entity_feature = np.zeros((num_entity, 2, num_relation))
    for triplet in triplets:
        h_id = triplet[0]
        t_id = triplet[1]
        r_id = triplet[2]
        entity_feature[h_id][0][r_id] += 1
        entity_feature[t_id][1][r_id] += 1
    return torch.IntTensor(entity_feature)