from random import sample
import tqdm


def write_triples(f, triplets):
    f.write("%d\n" % len(triplets))
    for triplet in triplets:
        f.write("%d %d %d\n" % (triplet[0], triplet[1], triplet[2]))
        

def write_entities(f, entities):
    f.write("%d\n" % len(entities))
    for e in entities:
        f.write("%d\n" % e)
        

# 读取所有实体
f_entities = open('./entity2id.txt', 'r')
entities = f_entities.readlines()[1: ]
entities = [int(e.split()[1]) for e in entities]
# 读取所有训练数据
f_train_triplets = open('./train2id.txt')
train_triplets = f_train_triplets.readlines()[1: ]
for i in range(len(train_triplets)):
    train_triplets[i] = list(map(int, train_triplets[i].split()))


SEED = 0.1


entities_OOKB = sample(entities, int(len(entities) * SEED))


# 训练数据list，测试数据list
train_triplets_OOKB = list()
test_triplets_OOKB = list()


# 遍历列表，筛选所有含有OOKB的entity
print("Searching the OOKB triplets...")
for triplet in train_triplets:
    h = triplet[0]
    r = triplet[2]
    t = triplet[1]
    if h in entities_OOKB or t in entities_OOKB:
        test_triplets_OOKB.append(triplet)
    else:
        train_triplets_OOKB.append(triplet)
print("There are %d OOKB train triplets and %d OOKB test tripets" % (len(train_triplets_OOKB), len(test_triplets_OOKB)))


# 从test中筛选出所有头实体为OOKB，尾实体为OOKB与bothOOKB的triplet
print("Spliting head tail both triplets...")
head_OOKB = list()
tail_OOKB = list()
both_OOKB = list()
for triplet in test_triplets_OOKB:
    h = triplet[0]
    r = triplet[2]
    t = triplet[1]
    if h in entities_OOKB and t not in entities_OOKB:
        head_OOKB.append(triplet)
    elif h not in entities_OOKB and t in entities_OOKB:
        tail_OOKB.append(triplet)
    elif h in entities_OOKB and t in entities_OOKB:
        both_OOKB.append(triplet)
print("Split finished.")


# 写文件
f_OOKB_entities = open("./%d/OOKB_entities.txt" % int(SEED * 100), 'w')
f_OOKB_train2id = open("./%d/train2id.txt" % int(SEED * 100), 'w')
f_OOKB_test2id = open("./%d/test2id.txt" % int(SEED *100), 'w')
f_OOKB_head = open("./%d/OOKB_head.txt" % int(SEED *100), 'w')
f_OOKB_tail = open("./%d/OOKB_tail.txt" % int(SEED *100), 'w')
f_OOKB_both = open("./%d/OOKB_both.txt" % int(SEED *100), 'w')
write_entities(f_OOKB_entities, entities_OOKB)
write_triples(f_OOKB_train2id, train_triplets_OOKB)
write_triples(f_OOKB_test2id, test_triplets_OOKB)
write_triples(f_OOKB_head, head_OOKB)
write_triples(f_OOKB_tail, tail_OOKB)
write_triples(f_OOKB_both, both_OOKB)
f_OOKB_entities.close()
f_OOKB_train2id.close()
f_OOKB_test2id.close()
f_OOKB_head.close()
f_OOKB_tail.close()
f_OOKB_both.close()
        
            