import numpy as np
from matplotlib import pyplot as plt
import os
import sys


def draw(x, y, model_name, dataset, store_path, if_show=False):
    fig = plt.figure()
    x = np.array(x)
    y = np.array(y)
    plt.title("{}\ntrain history on {}".format(model_name, dataset))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, y)
    
    batch_size = len(x) // 10
    batch_size = batch_size if batch_size > 0 else 1
    for i in range((len(x) // batch_size)):
        j = i * batch_size
        plt.text(x[j], y[j], "%.2f" % y[j])
    plt.text(x[-1], y[-1], "%.2f" % y[-1])
    plt.savefig(store_path)
    
    if if_show:
        plt.show()


def write_record(train_record, test_record, path):
    # record the train and test data
    f_record = open(path, 'a')
    f_record.write("===================================================================\n")
    for key, value in train_record.items():
        f_record.write("{}: {}\n".format(key, value))
    f_record.write("-------------------------------------------------------------------\n")
    for key, value in test_record.items():
        f_record.write("{}: {}\n".format(key, value))
    f_record.write("===================================================================")
    f_record.write("\n\n\n")
    f_record.close()


def write_feature_OOKB_record(train_record, test_record_1, test_record_2, path):
    # record the train data
    f_record = open(path, 'a')
    f_record.write("===================================================================\n")
    for key, value in train_record.items():
        f_record.write("{}: {}\n".format(key, value))
    f_record.write("-------------------------------------------------------------------\n")
    for key, value in test_record_1.items():
        f_record.write("{}: {}\n".format(key, value))
    f_record.write("-------------------------------------------------------------------\n")
    for key, value in test_record_2.items():
        f_record.write("{}: {}\n".format(key, value))
    f_record.write("===================================================================")
    f_record.write("\n\n\n")
    f_record.close()
    
    
    
def print_train_parameters(model_name, dim, p_norm, margin, learning_rate, nbatch, neg_ent):
    sys.stdout.write("Train {}\n".format(model_name))
    print("Parameter setting:\ndim: {}, p_norm: {}, margin: {}, learning_rate: {}, nbatch: {}, neg_ent: {}".format(
        dim, p_norm, margin, learning_rate, nbatch, neg_ent))