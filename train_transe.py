import time
import openke
import n
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.module.strategy import Feature
from openke.module.tool import Tools
from openke.data import TrainDataLoader, TestDataLoader


MODEL = 'transe'
CHECK_TIME = time.strftime("%Y%m%d%H%M")
MODEL_NAME = "{}_{}".format(MODEL, CHECK_TIME)
DATASET = "./benchmarks/FB15K237/"	# 数据集
DIM = 50	# 维度数
PNORM = 1	# 范数
MARGIN = 1.0	# transe距离转移margin
LEARNING_RATE = 0.005	# 学习率
EPOCH = 1	# 训练次数
NBATCH = 100	# 分多少个batch
NEG_ENT = 2	# 每个实体负采样数


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
Tools.draw(np.arange(0, EPOCH), trainer.loss_history, "Loss history", "Epoch", "Loss", 
          "./checkpoint/{}/{}.png".format(MODEL, MODEL_NAME))


# test the model
transe.load_checkpoint('./checkpoint/{}/{}.ckpt'.format(MODEL, MODEL_NAME))
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)


# record the train and test data
f_record = open('./checkpoint/{}/records.txt'.format(MODEL), 'a')
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
f_record.write("===================================================================\n")
for key, value in train_record.items():
    f_record.write("{}: {}\n".format(key, value))
f_record.write("-------------------------------------------------------------------\n")
f_record.write("test result(fliter):\n")
f_record.write("mrr: {}, mr: {}, hit10:{}, hit3:{}, hit1:{}\n".format(mrr, mr, hit10, hit3, hit1))
f_record.write("===================================================================")
f_record.write("\n\n")
f_record.close()