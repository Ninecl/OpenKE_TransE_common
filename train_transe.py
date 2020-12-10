import time
import openke
import numpy as np
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.module.tool import Feature
from openke.module.tool import Tools
from openke.data import TrainDataLoader, TestDataLoader


MODEL = 'transe'
CHECK_TIME = time.strftime("%Y%m%d%H%M")
MODEL_NAME = "{}_{}".format(MODEL, CHECK_TIME)
DATASET = "FB15K237"	# 数据集
DATAPATH = "./benchmarks/{}/".format(DATASET)	# 数据集路径
DIM = 200	# 维度数
PNORM = 1	# 范数
MARGIN = 5.0	# transe距离转移margin
LEARNING_RATE = 1.0	# 学习率
EPOCH = 1000	# 训练次数
NBATCH = 100	# batch_size大小
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

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = DIM, 
	p_norm = PNORM, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = MARGIN),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = EPOCH, alpha = LEARNING_RATE, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))


# draw the loss picture
Tools.draw(np.arange(0, EPOCH), trainer.loss_history, MODEL_NAME, DATASET, 
          "./checkpoint/{}/{}.png".format(MODEL, MODEL_NAME))


# test the model
transe.load_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)


# draw the loss picture
Tools.draw(np.arange(0, EPOCH), trainer.loss_history, MODEL_NAME, DATASET, 
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
