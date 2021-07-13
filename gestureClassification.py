import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv1D, Conv2D, Flatten,\
    Dense,Dropout, Input, Lambda,MaxPooling2D,\
    concatenate,BatchNormalization,MaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import random
from Preprocess.gestureDataLoader import gestureDataLoader
import numpy as np
import os
import scipy.io as sio
import re
from SiameseNetworkWithTripletLoss import SiamesNetworkTriplet_2
dataDir = 'D:/OneShotGestureRecognition/20181115/'
embedding = SiamesNetworkTriplet_2(batch_size=32,data_dir=dataDir,lr = 0.001)
network = embedding.build_embedding_network()
input = Input(embedding.input_shape,name='data input')
encoded_model = network(input)
optimizer = SGD(
        lr=0.0001,
        # lr_multipliers=learning_rate_multipliers,
        momentum=0.5 )
output = Dense(units = 6, activation= 'softmax')(encoded_model)
model = Model(inputs = input,outputs = output )
model.compile(loss = 'categorical_crossentropy',optimizer=optimizer,metrics = 'acc')
model.summary()


# load data
def loadData(dataDir = dataDir):
    fileName = os.listdir(dataDir)
    data = []
    labels = []
    for name in fileName:
        path = os.path.join(dataDir,name)
        data.append(sio.loadmat(path)['csiAmplitude'])
        gestureMark = int(re.findall( r'\d+\b', name )[ 1 ]) - 1
        labels.append(tf.keras.utils.to_categorical(gestureMark,num_classes=6))
    return np.asarray(data),np.asarray(labels)
data,labels = loadData()
X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.1, random_state=27 )
history = model.fit(X_train, y_train,validation_split=0.1, epochs=200)