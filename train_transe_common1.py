import time
import openke
import torch
import numpy as np
from openke.config import Trainer, Tester
from openke.module.model import TransE_common1
from openke.module.model import TransE_with_feature
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.module.strategy import NegativeSampling_common
from openke.module.strategy import Feature
from openke.module.tool import Tools
from openke.data import TrainDataLoader, TestDataLoader


MODEL = 'transe_common1'
CHECK_TIME = time.strftime("%Y%m%d%H%M")
MODEL_NAME = "{}_{}".format(MODEL, CHECK_TIME)
DATASET = "./benchmarks/FB15K237/"	# 数据集
DIM = 100	# 维度数
PNORM = 1	# 范数
MARGIN = 1.0	# transe距离转移margin
LEARNING_RATE = 0.005	# 学习率
EPOCH = 1	# 训练次数
NBATCH = 100	# 分多少个batch
NEG_ENT = 1	# 每个实体负采样数


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = DATASET, 
	nbatches = NBATCH,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = NEG_ENT,
	neg_rel = 0)


# dataloader for test
test_dataloader = TestDataLoader(DATASET, "link")


# get the feature of the entities
print("Caculate the feature of entities")
f_train_triples = open("{}/train2id.txt".format(DATASET), 'r')
train_triples = f_train_triples.readlines()[1: ]
for i in range(len(train_triples)):
    train_triples[i] = list(map(int, train_triples[i].split()))
entities_feature = Feature.get_entity_feature(train_dataloader.get_ent_tot(), 
                                              train_dataloader.get_rel_tot(), train_triples)


# ===============================================================================
# STEP1
# caculate the common feature of transe
# ===============================================================================


# define the model
transe_common1 = TransE_common1(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
    entities_feature = entities_feature, 
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


"""# test the model
transe_common1.load_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))
tester = Tester(model = transe_common1, data_loader = test_dataloader, use_gpu = True)
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)


# draw the loss picture
Tools.draw(np.arange(0, EPOCH), trainer.loss_history, "Loss history", "Epoch", "Loss", 
          "./checkpoint/{}/{}.png".format(MODEL, MODEL_NAME))


# record the train and test data
record_path = './checkpoint/{}/records.txt'.format(MODEL)
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
test_record = {
    "MRR": mrr, 
    "MR": mr, 
    "Hit10": hit10,
    "Hit3": hit3, 
    "Hit1": hit1
}
Tools.write_record(train_record, test_record, record_path)"""


# =================================================================================
# STEP2
# store the common_embedding and release GPU memory
# caculate the transe_with_feature
# =================================================================================
MODEL = 'transe_with_feature'
MODEL_NAME = "{}_{}".format(MODEL, CHECK_TIME)


torch.cuda.empty_cache()


# define the model
transe_with_feature = TransE_with_feature(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
    entities_feature = entities_feature, 
    common_head_embeddings = common_head_embeddings,
    common_tail_embeddings = common_tail_embeddings, 
	dim = DIM, 
	p_norm = PNORM, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe_with_feature, 
	loss = MarginLoss(margin = MARGIN),
	batch_size = train_dataloader.get_batch_size()
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = EPOCH, alpha = LEARNING_RATE, use_gpu = True)
trainer.run()
transe_with_feature.save_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))


# test the model
transe_with_feature.load_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))
tester = Tester(model = transe_with_feature, data_loader = test_dataloader, use_gpu = True)
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)


# draw the loss picture
Tools.draw(np.arange(0, EPOCH), trainer.loss_history, "Loss history", "Epoch", "Loss", 
          "./checkpoint/{}/{}.png".format(MODEL, MODEL_NAME))


# record the train and test data
record_path = './checkpoint/{}/records.txt'.format(MODEL)
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
test_record = {
    "MRR": mrr, 
    "MR": mr, 
    "Hit10": hit10,
    "Hit3": hit3, 
    "Hit1": hit1
}
Tools.write_record(train_record, test_record, record_path)