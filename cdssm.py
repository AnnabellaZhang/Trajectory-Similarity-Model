path = "/data/zrb/Trajectory-data/dssm"
model = 'DNN-v1'


import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import keras
from utils import create_logger
import pandas as pd
import numpy as np
import six.moves.cPickle as pickle
from sklearn.utils import shuffle
import os


def print_and_log(string, logger):
    print(string)
    if logger:
        logger.info(string)

cur_time = time.strftime('%Y-%m-%d-%H-%M')
if not os.path.exists('output/' + model + '-' + cur_time):
    exp_name = cur_time
    out_path = 'output/'+ model + '-' + cur_time
    os.makedirs(out_path)
    logger = create_logger('./{}/logs'.format(out_path), exp_name)
    print_and_log('Creating folder: {}'.format(out_path), logger)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

f = open('./data/input_geo.pkl','rb')
data = pickle.load(f)
f.close()

train,test = train_test_split(data,test_size = 0.2,random_state=20,shuffle=True)
train,dev = train_test_split(train,test_size=0.125,random_state=20,shuffle=True)

print("train:{} dev:{} test:{}".format(train.shape,dev.shape,test.shape))

TRA_LENGTH = len(X1_train[0])
DNN1 = 32768
DNN2 = 2048
DNN3 = 128

from keras.layers import Dense,Dropout,Input,Flatten,Activation,Reshape
from keras.layers.merge import Dot
from keras.models import Model
from keras.optimizers import RMSprop,SGD,Adam,Adagrad
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping

def DnnModel():
    print_and_log("DNN Model",logger)
    input1 = Input(shape=(TRA_LENGTH,None),name="trajectory1")
    input2 = Input(shape=(TRA_LENGTH,None), name="trajectory2")
    # Shared Layer
    dnn1 = Dense(DNN1,activation='relu')
    dropout1 = Dropout(0.8)
    reshape1 = Reshape((DNN1, None))
    dnn2 = Dense(DNN2, activation='relu')
    dropout2 = Dropout(0.8)
    reshape2 = Reshape((DNN2, None))
    dnn3 = Dense(DNN3)
    reshape3 = Reshape((DNN3,None))
    output = Activation('softmax')
    #Model
    #trajectory1
    dnn1_1 = dnn1(input1)
    drop1_1 = dropout1(dnn1_1)
    reshape1_1 = reshape1(drop1_1)
    dnn2_1 = dnn2(reshape1_1)
    drop2_1 = dropout2(dnn2_1)
    reshape2_1 = reshape2(drop2_1)
    dnn3_1 = dnn3(reshape2_1)
    reshape3_1 = reshape3(dnn3_1)
    output1 = output(reshape3_1)
    #trajectory2
    dnn1_2 = dnn1(input2)
    drop1_2 = dropout1(dnn1_2)
    reshape1_2 = reshape1(drop1_2)
    dnn2_2 = dnn2(reshape1_2)
    drop2_2 = dropout2(dnn2_2)
    reshape2_2 = reshape2(drop2_2)
    dnn3_2 = dnn3(reshape2_2)
    reshape3_2 = reshape3(dnn3_2)
    output2 = output(reshape3_2)
    y = Dot(normalize=True)(output1,output2)

    model = Model(inputs=[input1,input2],output=y)
    opt = Adam(lr=1e-4)
    model.compile(loss='logloss',optimizer=opt,metrics="binary_crossentropy")
    return model

def get_callbacks(filepath, patience=2):
   es = EarlyStopping('val_loss', patience=20, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]

def main(train,test,dev):
    #X_train, Y_train = train[['tra1','tra2']], train['label']
    #X_dev, Y_dev = dev[['tra1','tra2']], dev['label']
    #X_test, Y_test = test[['tra1','tra2']], test['label']

    file_path = 'output/' + model + "-" + cur_time
    callbacks = get_callbacks(filepath=file_path, patience=10)
    train = shuffle(train)
    print_and_log("N_train: {}".format(len(train)), logger)
    #print_and_log("Steps_per_epoch: {}".format(math.ceil(len(X_train) / batch_size)), logger)


