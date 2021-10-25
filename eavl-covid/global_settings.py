""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

# #mean and std of cifar100 dataset
# CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)



WORK_DIR = './data'

#superparameters
POP_SIZE = 50
BATCH_SIZE = 128
LEARNING_RATE =1e-4
NUM_EPOCH  = 50
PROB_CROSSOVER = 0.9
generation = 1

NUM_BEST_INDIVIDUAL = 10
NUM_CROSSOVER = 30

#DNA parameters
NUM_HIDDEN_LAYERS = [5,15]
NUM_FULL_CONNECTION = [2,6]
OUT_CHANNELS = [50,100]
NUM_CLASSIFICATION = 100
FILTER_KERNEL_SIZE = [3]
NUM_FULL_CONNECTION_NERONS = [1000,5000]
KIND_OF_LAYERS = ['c', 'p', 'cn']

MODEL_PATH = './models'
MODEL_NAME = 'vgg19_bn.pth'


#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








