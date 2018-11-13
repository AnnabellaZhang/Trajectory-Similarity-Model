path = "/data/zrb/Trajectory-data/dssm"
model = 'DNN-v1'
EPOCH_NUM = 5
BATCH_SIZE = 1024

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
import math


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

train,test = train_test_split(data,test_size = 0.05,random_state=20,shuffle=True)
train,dev = train_test_split(train,test_size=0.05,random_state=20,shuffle=True)

print("train:{} dev:{} test:{}".format(train.shape,dev.shape,test.shape))

TRA_LENGTH = len(train['tra1'][0])
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
    input1 = Input(shape=(TRA_LENGTH,),name="trajectory1")
    input2 = Input(shape=(TRA_LENGTH,), name="trajectory2")
    # Shared Layer
    dnn1 = Dense(DNN1,activation='relu')#,input_shape=(TRA_LENGTH,None))
    dropout1 = Dropout(0.8)
    reshape1 = Reshape((DNN1,))
    dnn2 = Dense(DNN2, activation='relu')
    dropout2 = Dropout(0.8)
    reshape2 = Reshape((DNN2,))
    dnn3 = Dense(DNN3)
    reshape3 = Reshape((DNN3,))
    output = Activation('softmax')
    #Model
    #trajectory1
    print("input1:{}".format(input1.shape))
    dnn1_1 = dnn1(input1)
    drop1_1 = dropout1(dnn1_1)
    reshape1_1 = reshape1(drop1_1)
    print("reshape1_1:{}".format(reshape1_1.shape))
    dnn2_1 = dnn2(reshape1_1)
    drop2_1 = dropout2(dnn2_1)
    reshape2_1 = reshape2(drop2_1)
    print("reshape2_1:{}".format(reshape2_1.shape))
    dnn3_1 = dnn3(reshape2_1)
    reshape3_1 = reshape3(dnn3_1)
    output1 = output(reshape3_1)
    print("output1:{}".format(output1.shape))
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
    y = Dot(axes=1,normalize=True)([output1,output2])


    model = Model(inputs=[input1,input2],output=y)
    opt = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["binary_crossentropy"])
    return model

def get_callbacks(filepath, patience=2):
   es = EarlyStopping('val_loss', patience=20, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]

def main(train,test,dev):
    #print("Prepare dev set and test set.")
    #dev_tra1 = np.load("./data/dev_tra1.npy") #np.array([np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in dev['tra1']])
    #dev_tra2 = np.load("./data/dev_tra2.npy") #np.array([np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in dev['tra2']])
    #dev_y = np.load("./data/dev_y.npy") #np.array([np.array([y]).reshape(1, ) for y in dev['label']])
    #test_tra1 = np.load("./data/test_tra1.npy") #np.array([np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in test['tra1']])
    #test_tra2 = np.load("./data/test_tra2.npy") #np.array([np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in test['tra2']])
    #test_y = np.load("./data/test_y.npy") #np.array([np.array([y]).reshape(1, ) for y in test['label']])
    #TEST_NUM = len(test)
    y_pred = []
    cur_model = DnnModel()
    print_and_log("N_train: {}".format(len(train)), logger)
    print_and_log("Batch_size: {}".format(BATCH_SIZE), logger)
    for i in range(1,EPOCH_NUM+1):
        print_and_log("Epoch: {}".format(i), logger)
        STEP = math.ceil(len(train) / BATCH_SIZE)
        print_and_log("Steps_per_epoch: {}".format(STEP), logger)
        train = shuffle(train)
        test_score = 0.0
        for j in range(1,STEP+1):
            tra1 = np.array([np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in train[(j-1)*BATCH_SIZE:j*BATCH_SIZE]['tra1']])
            tra2 = np.array([np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in train[(j - 1) * BATCH_SIZE:j * BATCH_SIZE]['tra2']])
            y = np.array([np.array([y]).reshape(1, ) for y in train[(j - 1) * BATCH_SIZE:j * BATCH_SIZE]['label']])
            callbacks = get_callbacks(filepath=file_path, patience=10)
            history =cur_model.fit([tra1,tra2],y,epochs=1,verbose=1,callbacks=callbacks,shuffle=True)
            score = cur_model.evaluate([tra1,tra2],y,verbose=1)
            print_and_log('epoch {}, batch {}, Train loss: {}'.format(i,j, score[0]), logger)
            #score = cur_model.evaluate([dev_tra1,dev_tra2],dev_y,verbose=1)
            #print_and_log('epoch {}, batch {}, Dev loss: {}'.format(i,j, score[0]), logger)
            #score = cur_model.evaluate([test_tra1,test_tra2],test_y,verbose=1)
            #print_and_log('epoch {}, batch {}, Test loss: {}'.format(i,j, score[0]), logger)
            if j == STEP:
                test_score = score[0]
        file_path = 'output/' + model + "-" + cur_time + "/{}_model_weights.hdf5".format(i)
        print_and_log('Epoch: {}, Test loss: {}'.format(i, test_score), logger)
        #tmp = cur_model.predict([test_tra1,test_tra2],verbose=1)
        #y_pred.append(tmp)
    return np.array(y_pred)

preds = main(train,test,dev)
np.save('output/' + model + "-" + cur_time + "/preds.npy",preds)



