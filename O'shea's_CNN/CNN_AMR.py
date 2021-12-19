import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import random, sys, keras
import _pickle as cPickle
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import math
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
#from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
#from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import  tqdm
from keras.layers import Input, Reshape, ZeroPadding2D, Conv2D, Dropout, Flatten, BatchNormalization, Dense, \
    Activation, \
    MaxPooling2D, AlphaDropout, GlobalAveragePooling2D, multiply, CuDNNLSTM, CuDNNGRU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


Xd = cPickle.load(open("C://Users//lls//Desktop//调制识别//RML2016.10a_dict.pkl",'rb'), encoding='bytes')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
def to_onehot(yy):
    yy = list(yy)
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

np.random.seed(2016)
n_examples = X.shape[0]
split_size = 0.8
train_idx = []
test_idx = []
for i in tqdm(range(n_examples)):
    if i+1000 <= n_examples:
        if i % 1000 == 0 :
            ntrain_idx = np.random.choice(range(i,i+1000),size=int(1000*split_size),replace=False)
            ntest_idx = list(set(range(i,i+1000))-set(ntrain_idx))
            train_idx.append(ntrain_idx)
            test_idx.append(ntest_idx)
train_idx = np.array(train_idx).reshape(-1)
test_idx = np.array(test_idx).reshape(-1)
X_train = X[train_idx]
X_test = X[test_idx]

Y_train = list(map(lambda x : mods.index(lbl[x][0]),train_idx))
Y_test = list(map(lambda x : mods.index(lbl[x][0]),test_idx))
X_train,X_valid,Y_train,Y_valid = train_test_split(X_train,Y_train,test_size = 0.125)

#稀疏编码
Y_train = to_onehot(Y_train)
Y_test = to_onehot(Y_test)
Y_valid = to_onehot(Y_valid)

in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods

import itertools


def plot_confusion_matrix(cm, title='', cmap=plt.cm.Blues, labels=[], snr=None, model_name=None):
    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), size=5, horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def CNN2(classes=11):
    dr = 0.5
    inputs = Input((2, 128))
    x = Reshape((2, 128, 1))(inputs)
    x = ZeroPadding2D((0, 2))(x)
    x = Conv2D(256, kernel_size=2,padding='same', activation="relu", name="conv1")(x)
    x = Dropout(dr)(x)
    x = ZeroPadding2D((0, 2))(x)
    x = Conv2D(80, kernel_size=2,padding="same", activation="relu", name="conv2")(x)
    x = Dropout(dr)(x)
    x = Flatten()(x)
    # x = Dense(512, activation='relu', init='he_normal', name="dense1")(x)
    # x = Dropout(dr)(x)
    x = Dense(256, activation='relu', name="dense2")(x)
    x = Dropout(dr)(x)
    # x = Dense(64, activation='relu', name="dense3")(x)
    # x = Dropout(dr)(x)
    x = Dense(len(classes), name="dense4")(x)

    outputs = Activation('softmax')(x)
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = CNN2(classes=classes)
model.summary()

nb_epoch = 50     # number of epochs to train on
batch_size = 128  # training batch size

Model_name = 'CNN2.h5'
adam = tf.keras.optimizers.Adam(lr=0.001)
#adam=tf.keras.op
model.compile(optimizer=adam,
                  loss=['categorical_crossentropy'],
                  metrics=['categorical_accuracy'])
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_valid, Y_valid),
    callbacks = [
        keras.callbacks.ModelCheckpoint(Model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(Model_name)

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print (score)

acc = {}
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

    # plt.figure()
    # plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    plot_confusion_matrix(confnorm, labels=classes, title="",snr=snr,model_name=Model_name)
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("at SNR={}".format(snr), "Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)

plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)), marker='o')
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10A")
plt.ylim(([0,1]))
plt.grid()
plt.show()