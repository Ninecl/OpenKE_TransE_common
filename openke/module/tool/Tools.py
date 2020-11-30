import numpy as np
from matplotlib import pyplot as plt
import os

def draw(x, y, title, x_description, y_description, store_path, if_show=False):
    x = np.array(x)
    y = np.array(y)
    plt.title(title)
    plt.xlabel(x_description)
    plt.ylabel(y_description)
    plt.plot(x, y)
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