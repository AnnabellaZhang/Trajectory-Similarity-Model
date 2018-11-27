import sys
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

path = "/data/zrb/Trajectory-data/dssm"
model = 'DNN-v2'
BATCH_SIZE = 256

# Input: input path & output path
EPOCH_NUM,gpu,LOSS,PATIENCE,LAST_MODEL = 30,"0","mean_squared_error",5,""

if len(sys.argv) != 6:
	print("Parameters: EPOCH_NUM,CUDA_DIVICE,LOSS,PATIENCE,LAST_MODEL")
	sys.exit()
else:
    EPOCH_NUM = int(sys.argv[1].strip())
    gpu = sys.argv[2].strip()
    LOSS = sys.argv[3].strip()
    PATIENCE = int(sys.argv[4].strip())
    LAST_MODEL = sys.argv[5].strip()

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
print_and_log('EPOCH_NUM: {}'.format(EPOCH_NUM), logger)
print_and_log('CUDA_DIVICE: {}'.format(gpu), logger)
print_and_log('LOSS: {}'.format(LOSS), logger)
print_and_log('PATIENCE: {}'.format(PATIENCE), logger)
print_and_log('LAST_MODEL:{}'.format(LAST_MODEL),logger)
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

#data = data[0:10000]

train,test = train_test_split(data,test_size = 0.05,random_state=20,shuffle=True)
train,dev = train_test_split(train,test_size=0.05,random_state=20,shuffle=True)

print("train:{} dev:{} test:{}".format(train.shape,dev.shape,test.shape))

TRA_LENGTH = len(train['tra1'][0])
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


def get_callbacks(filepath):
    es = EarlyStopping(LOSS, patience=PATIENCE, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True,monitor=LOSS,mode='min')
    return [es, msave]

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

def counter():
    i = 1
    while 1:
        yield i
        i += 1

def main(train,test,dev):
    print("Prepare dev set and test set.")
    dev_tra1 = np.load("./data/dev_tra1.npy")  #  np.array([np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in dev['tra1']])
    dev_tra2 = np.load("./data/dev_tra2.npy") #np.array([np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in dev['tra2']])
    dev_y = np.load("./data/dev_y.npy") #np.array([np.array([y]).reshape(1, ) for y in dev['label']])
    test_tra1 = np.load("./data/test_tra1.npy") #np.array([np.array(tra1).reshape(TRA_LENGTH, ) for tra1 in test['tra1']])
    test_tra2 = np.load("./data/test_tra2.npy") #np.array([np.array(tra2).reshape(TRA_LENGTH, ) for tra2 in test['tra2']])
    test_y = np.load("./data/test_y.npy") #np.array([np.array([y]).reshape(1, ) for y in test['label']])
    #TEST_NUM = len(test)
    STEP = math.ceil(len(train) / BATCH_SIZE)
    cur_model = DnnModel()
    cur_model.load_weights('./output/' + LAST_MODEL + '/aug_model_weights.hdf5')
    print_and_log("Load Model: {}".format(LAST_MODEL + "/aug_model_weights.hdf5"), logger)
    print_and_log("N_train: {}".format(len(train)), logger)
    print_and_log("Batch_size: {}".format(BATCH_SIZE), logger)
    print_and_log("Steps_per_epoch: {}".format(STEP), logger)
    data_flow = generate_data(train)
    c = counter()
    file_path = 'output/' + model + '-' + cur_time + "/aug_model_weights.hdf5" #% next(c)
    callbacks = get_callbacks(filepath=file_path)
    history = cur_model.fit_generator(
        data_flow,
        steps_per_epoch=STEP,
        epochs=EPOCH_NUM,
        shuffle=True,
        use_multiprocessing=True,
        verbose=1,
        validation_data=([dev_tra1,dev_tra2],dev_y),
        callbacks=callbacks)
    cur_model.load_weights(filepath=file_path)
    score = cur_model.evaluate_generator(data_flow,steps=STEP)
    print_and_log('Train loss: {}'.format(score[0]), logger)
    print_and_log('Train hinge: {}'.format(score[2]), logger)
    score = cur_model.evaluate([dev_tra1,dev_tra2],dev_y,verbose=1)
    print_and_log('Val loss: {}'.format(score[0]), logger)
    print_and_log('Val hinge: {}'.format(score[2]), logger)
    score = cur_model.evaluate([test_tra1,test_tra2],test_y,verbose=1)
    print_and_log('Test loss: {}'.format(score[0]), logger)
    print_and_log('Test hinge: {}'.format(score[2]), logger)
    y_pred = cur_model.predict([test_tra1,test_tra2])
    y_pred = y_pred.reshape(y_pred.shape[0])
    return [history.history,np.array(y_pred)]

[history,preds] = main(train,test,dev)
np.save('output/' + model + "-" + cur_time + "/preds.npy",preds)
with open('output/' + model + "-" + cur_time + "/history.json",'a') as outfile:
    json.dump(history,outfile,ensure_ascii=False)
    outfile.write('\n')
