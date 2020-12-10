import time
import openke
import torch
import numpy as np
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.model import TransE_common1
from openke.module.model import TransE_with_feature
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.module.strategy import NegativeSampling_common
from openke.module.tool import Feature
from openke.module.tool import Tools
from openke.data import TrainDataLoader, TestDataLoader


MODEL = 'transe_common1_OOKB_10'
CHECK_TIME = time.strftime("%Y%m%d%H%M")
MODEL_NAME = "{}_{}".format(MODEL, CHECK_TIME)
DATASET = "FB15K237_OOKB/10"	# 数据集
DATAPATH = "./benchmarks/{}/".format(DATASET)	# 数据集路径
DIM = 200	# 维度数
PNORM = 1	# 范数
MARGIN = 5.0	# transe距离转移margin
LEARNING_RATE = 0.1	# 学习率
EPOCH = 1000	# 训练次数
NBATCH = 100	# batch_size为多少
NEG_ENT = 25	# 每个实体负采样数


# parameters to screen
Tools.print_train_parameters(MODEL_NAME, DIM, PNORM, MARGIN, LEARNING_RATE, NBATCH, NEG_ENT)


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = DATAPATH, 
	nbatches = NBATCH,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = NEG_ENT,
	neg_rel = 0)


# dataloader for test
test_dataloader = TestDataLoader(DATAPATH, "link")


# get the feature of the entities
print("Caculate the feature of entities")
f_train_triples = open("{}/train2id.txt".format(DATAPATH), 'r')
train_triples = f_train_triples.readlines()[1: ]
for i in range(len(train_triples)):
    train_triples[i] = list(map(int, train_triples[i].split()))
entities_original_feature = Feature.get_entity_feature(train_dataloader.get_ent_tot(), 
                                              train_dataloader.get_rel_tot(), train_triples)
print("Caculate the OOKB feature...")
f_OOKB_triplets = open("{}/test2id.txt".format(DATAPATH), 'r')
test_triples = f_OOKB_triplets.readlines()[1: ]
for i in range(len(test_triples)):
    test_triples[i] = list(map(int, test_triples[i].split()))
entities_OOKB_feature = Feature.get_entity_feature(train_dataloader.get_ent_tot(), 
                                                   train_dataloader.get_rel_tot(), test_triples)


# ===============================================================================
# STEP1
# caculate the common feature of transe
# ===============================================================================


# define the model
transe_common1 = TransE_common1(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
    entities_feature = entities_original_feature, 
	dim = DIM, 
	p_norm = PNORM, 
	norm_flag = True)


# define the loss function
model = NegativeSampling_common(
	model = transe_common1, 
	loss = MarginLoss(margin = MARGIN),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = EPOCH, alpha = LEARNING_RATE, use_gpu = True)
trainer.run()
transe_common1.save_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))


# load common features and caculate entities feature embeddings
common_head_embeddings = transe_common1.common_head_embeddings.weight.data.cpu()
common_tail_embeddings = transe_common1.common_tail_embeddings.weight.data.cpu()
entities_feature = entities_original_feature + entities_OOKB_feature
entities_feature_weight = Feature.caculate_feature_weight(entities_feature)
entities_feature_embedding = Feature.pool_feature(entities_feature_weight, common_head_embeddings, common_tail_embeddings)
# release GPU memory
torch.cuda.empty_cache()


# test the model
transe_common1.load_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))
transe_with_feature = TransE_with_feature(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
    entities_feature = entities_feature, 
    common_head_embeddings = common_head_embeddings,
    common_tail_embeddings = common_tail_embeddings, 
    entities_feature_embedding = entities_feature_embedding, 
	dim = DIM, 
	p_norm = PNORM, 
	norm_flag = True)
tester = Tester(model = transe_with_feature, data_loader = test_dataloader, use_gpu = True)
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)


# draw the loss picture
Tools.draw(np.arange(0, EPOCH), trainer.loss_history, MODEL_NAME, DATASET, 
          "./checkpoint/{}/{}.png".format(MODEL, MODEL_NAME))


# record the train and test data
train_record = {
	"model_name": "{}.ckpt".format(MODEL_NAME), 
	"dim": "{}".format(DIM), 
	"nbatch": "{}".format(NBATCH), 
	"neg_sample": "{}".format(NEG_ENT), 
	"learning_rate": "{}".format(LEARNING_RATE), 
	"train_times": "{}".format(EPOCH), 
	"p_norm": "{}".format(PNORM), 
	"margin": "{}".format(MARGIN), 
	"date_set": "{}".format(DATASET)
}
test_record_1 = {
    "model_name": MODEL_NAME,
    "MRR": mrr, 
    "MR": mr, 
    "Hit10": hit10,
    "Hit3": hit3, 
    "Hit1": hit1
}


# =================================================================================
# STEP2
# use transe to predict the model directly
# =================================================================================
TEST_MODEL = 'transe'
MODEL_NAME = "transe_202012031931"


# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = DIM, 
	p_norm = PNORM, 
	norm_flag = True)


# test the model
transe.load_checkpoint('./checkpoint/{}/{}.ckpt'.format(TEST_MODEL, MODEL_NAME))
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)


# record the train and test data
record_path = './checkpoint/{}/records.txt'.format(MODEL)
test_record_2 = {
    "model_name": MODEL_NAME,
    "MRR": mrr, 
    "MR": mr, 
    "Hit10": hit10,
    "Hit3": hit3, 
    "Hit1": hit1
}
Tools.write_feature_OOKB_record(train_record, test_record_1, test_record_2, record_path)