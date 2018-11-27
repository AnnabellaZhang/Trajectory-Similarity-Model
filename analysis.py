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
import json
import sys

# Input: input path & output path
MODEL_NAME,gpu,MODEL = "DNN-v1-2018-11-19-15-11",'0',"DNN-v2"

if len(sys.argv) != 4:
        print("Parameters: MODEL_NAME,CUDA_DIVICE,LOSS,PATIENCE")
        sys.exit()
else:
    MODEL_NAME = sys.argv[1].strip()
    gpu = sys.argv[2].strip()
    MODEL = sys.argv[3].strip()


model_path = "./output/" + MODEL_NAME + "/"
data_path = "./data/"
EPOCH_NUM = 30
BATCH_SIZE = 256
LOSS = 'mean_squared_error'
print(model_path)


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# limit GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
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
if MODEL == "DNN-v1":
    DNN1 = 4096
    DNN2 = 1024
    DNN3 = 128
if MODEL == "DNN-v2":
    DNN1 = 8192
    DNN2 = 2048
    DNN3 = 512
    DNN4 = 256
    DNN5 = 128

from keras.layers import Dense,Dropout,Input,Flatten,Activation,Reshape,Lambda
from keras.layers.merge import Dot
from keras.models import Model
from keras.optimizers import RMSprop,SGD,Adam,Adagrad
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras import backend as K

def DnnModel1():
    input1 = Input(shape=(TRA_LENGTH,),name="trajectory1")
    input2 = Input(shape=(TRA_LENGTH,), name="trajectory2")
    # Shared Layer
    dnn1 = Dense(DNN1,activation='relu')#,input_shape=(TRA_LENGTH,None))
    dropout1 = Dropout(0.8)
    reshape1 = Reshape((DNN1,))
    dnn2 = Dense(DNN2, activation='relu')
    dropout2 = Dropout(0.8)
    reshape2 = Reshape((DNN2,))
    dnn3 = Dense(DNN3,activation='relu')
    output = Reshape((DNN3,))
    #output = Activation('softmax')
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
    output1 = output(dnn3_1)
    print("output1:{}".format(output1.shape))
    #trajectory2
    dnn1_2 = dnn1(input2)
    drop1_2 = dropout1(dnn1_2)
    reshape1_2 = reshape1(drop1_2)
    dnn2_2 = dnn2(reshape1_2)
    drop2_2 = dropout2(dnn2_2)
    reshape2_2 = reshape2(drop2_2)
    dnn3_2 = dnn3(reshape2_2)
    output2 = output(dnn3_2)
    #y = Dot(axes=1,normalize=True,name='y')([output1,output2])
    def cosine(inputs):
        x,y = inputs
        x_norm = K.sqrt(K.sum(K.square(x),axis=1,keepdims=True))
        y_norm = K.sqrt(K.sum(K.square(y),axis=1,keepdims=True))
        x_y = K.sum(x*y,axis=1,keepdims=True)
        ans = x_y / (x_norm * y_norm)
        return ans
    y = Lambda(cosine,name='y')([output1,output2])
    print("y:{}".format(y.shape))
    model = Model(inputs=[input1,input2],output=y)
    opt = Adam(lr=1e-4)
    model.compile(loss='mean_squared_error',optimizer=opt,metrics=["mean_squared_error","hinge"])
    return model

def DnnModel2():
    input1 = Input(shape=(TRA_LENGTH,),name="trajectory1")
    input2 = Input(shape=(TRA_LENGTH,), name="trajectory2")
    # Shared Layer
    dnn1 = Dense(DNN1,activation='relu')#,input_shape=(TRA_LENGTH,None))
    dropout1 = Dropout(0.8)
    reshape1 = Reshape((DNN1,))
    dnn2 = Dense(DNN2, activation='relu')
    dropout2 = Dropout(0.8)
    reshape2 = Reshape((DNN2,))
    dnn3 = Dense(DNN3,activation='relu')
    dropout3 = Dropout(0.8)
    reshape3 = Reshape((DNN3,))
    dnn4 = Dense(DNN4,activation='relu')
    dropout4 = Dropout(0.8)
    reshape4 = Reshape((DNN4,))
    dnn5 = Dense(DNN5,activation='relu')
    output = Reshape((DNN5,))
    #output = Activation('softmax')
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
    drop3_1 = dropout3(dnn3_1)
    reshape3_1 = reshape3(drop3_1)
    print("reshape3_1:{}".format(reshape3_1.shape))
    dnn4_1 = dnn4(reshape3_1)
    drop4_1 = dropout4(dnn4_1)
    reshape4_1 = reshape4(drop4_1)
    print("reshape4_1:{}".format(reshape4_1.shape))
    dnn5_1 = dnn5(reshape4_1)
    output1 = output(dnn5_1)
    print("output1:{}".format(output1.shape))
    #trajectory2
    dnn1_2 = dnn1(input2)
    drop1_2 = dropout1(dnn1_2)
    reshape1_2 = reshape1(drop1_2)
    dnn2_2 = dnn2(reshape1_2)
    drop2_2 = dropout2(dnn2_2)
    reshape2_2 = reshape2(drop2_2)
    dnn3_2 = dnn3(reshape2_2)
    drop3_2 = dropout3(dnn3_2)
    reshape3_2 = reshape3(drop3_2)
    dnn4_2 = dnn4(reshape3_2)
    drop4_2 = dropout4(dnn4_2)
    reshape4_2 = reshape4(drop4_2)
    dnn5_2 = dnn5(reshape4_2)
    output2 = output(dnn5_2)
    #y = Dot(axes=1,normalize=True,name='y')([output1,output2])
    def cosine(inputs):
        x,y = inputs
        x_norm = K.sqrt(K.sum(K.square(x),axis=1,keepdims=True))
        y_norm = K.sqrt(K.sum(K.square(y),axis=1,keepdims=True))
        x_y = K.sum(x*y,axis=1,keepdims=True)
        ans = x_y / (x_norm * y_norm)
        return ans
    y = Lambda(cosine,name='y')([output1,output2])
    print("y:{}".format(y.shape))
    model = Model(inputs=[input1,input2],output=y)
    opt = Adam(lr=1e-4)
    model.compile(loss=LOSS,optimizer=opt,metrics=["mean_squared_error","hinge"])
    return model


def generate_data(data):
    STEP = math.ceil(len(data) / BATCH_SIZE)
    while 1:
        data = shuffle(data)
        for j in range(1, STEP + 1):
            tra1 = np.array(
                [np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in data[(j - 1) * BATCH_SIZE:j * BATCH_SIZE]['tra1']])
            tra2 = np.array(
                [np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in data[(j - 1) * BATCH_SIZE:j * BATCH_SIZE]['tra2']])
            y = np.array([np.array([y]).reshape(1, ) for y in data[(j - 1) * BATCH_SIZE:j * BATCH_SIZE]['label']])
            yield ({'trajectory1':tra1,'trajectory2':tra2},{'y':y})

def main(train,test,dev):
    '''
    print("Prepare dev set and test set.")
    dev_tra1 = np.load("./data/dev_tra1.npy")  #  np.array([np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in dev['tra1']])
    dev_tra2 = np.load("./data/dev_tra2.npy") #np.array([np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in dev['tra2']])
    dev_y = np.load("./data/dev_y.npy") #np.array([np.array([y]).reshape(1, ) for y in dev['label']])
    test_tra1 = np.load("./data/test_tra1.npy") #np.array([np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in test['tra1']])
    test_tra2 = np.load("./data/test_tra2.npy") #np.array([np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in test['tra2']])
    test_y = np.load("./data/test_y.npy") #np.array([np.array([y]).reshape(1, ) for y in test['label']])
    '''
    #TEST_NUM = len(test)
    STEP = math.ceil(len(train) / BATCH_SIZE)
    if MODEL == "DNN-v1":
        cur_model = DnnModel1()
    if MODEL == 'DNN-v2':
        cur_model = DnnModel2()
    print("N_train: {}".format(len(train)), )
    print("Batch_size: {}".format(BATCH_SIZE))
    print("Steps_per_epoch: {}".format(STEP))
    data_flow = generate_data(train)
    file_path = model_path + "aug_model_weights.hdf5"
    cur_model.load_weights(filepath=file_path)
    tmp = next(data_flow)
    tra1,tra2,y = tmp[0]['trajectory1'],tmp[0]['trajectory2'],tmp[1]['y'].astype('float32')
    score = cur_model.evaluate([tra1, tra2], y, verbose=1)
    print('Train loss: {}'.format(score[0]))
    y_pred = cur_model.predict([tra1,tra2])
    #y_pred = y_pred.reshape(y_pred.shape[0])
    return np.concatenate((y,y_pred),axis=1)

result = main(train,test,dev)
np.save(model_path + "analysis.npy",result)