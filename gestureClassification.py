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
from sklearn.metrics.pairwise import cosine_similarity
dataDir = 'D:/OneShotGestureRecognition/20181115/'
embedding = SiamesNetworkTriplet_2(batch_size=32,data_dir=dataDir,lr = 0.001)
network = embedding.build_embedding_network()
# network.add( Lambda( lambda x: K.l2_normalize( x, axis=-1 ) ) )
# input = Input(embedding.input_shape,name='data input')
input = Input([1600,7],name='data input')
encoded_model = network(input)
optimizer = tf.keras.optimizers.Adam(
        lr=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        # lr_multipliers=learning_rate_multipliers,
         )
output = Dense(units = 6, activation= 'softmax')(encoded_model)
model = Model(inputs = input,outputs = output )
model.compile(loss = 'categorical_crossentropy',optimizer=optimizer,metrics = 'acc')
network.summary()

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
X_train = X_train.reshape( np.shape( X_train )[ 0 ], X_train.shape[ 2 ], X_train.shape[ 1 ] )
history = model.fit(X_train, y_train,validation_split=0.1, epochs=1)
def reshapeData(x):
    x = x.reshape( np.shape( x )[ 0 ], x.shape[ 2 ], x.shape[ 1 ] )
    return x
def Testing( test_dir:str,embedding_model ):
    test_sample = 100
    nway_min = 2
    nway_max = 6
    test_acc = [ ]
    nway_list = [ ]
    for nway in range( nway_min, nway_max + 1 ):
        print( "Checking %d way accuracy...." % nway )
        correct_count = 0
        for _ in range( test_sample ):
            # Retrieving nway number of triplets and calculating embedding vector
            nway_anchor, nway_positive, _ = gestureDataLoader(  data_path =test_dir,
                                                                batch_size = nway ).getTripletTrainBatcher( )
            nway_anchor = reshapeData(nway_anchor)
            nway_positive = reshapeData(nway_positive)
            # support set, it has N different classes depending on the batch_size
            # nway_anchor has the same class with nway_positive at the same row
            nway_anchor_embedding = embedding_model.predict( nway_anchor )

            sample_index = random.randint( 0, nway - 1 )
            sample_embedding = embedding_model.predict( np.expand_dims( nway_positive[ sample_index ], axis=0 ) )
            # print(sample_index, nway_anchor_embedding.shape, sample_embedding.shape)
            # sim = K.sum( K.square( nway_anchor_embedding - sample_embedding ), axis = 1 )
            # using cosine_similarity
            sim = cosine_similarity( nway_anchor_embedding, np.expand_dims( sample_embedding[sample_index ], axis = 0 ) )
            if np.argmax( sim ) == sample_index:
                correct_count += 1
        nway_list.append( nway )
        acc = (correct_count / test_sample) * 100.
        test_acc.append( acc )
        print( "Accuracy %.2f" % acc )
Testing(test_dir = 'D:/OneShotGestureRecognition/20181115/',embedding_model = network)

# Output for sipecific layer
# desiredLayers = [15]
# desiredOutputs = [network.layers[i].output for i in desiredLayers]
# newModel = Model(network.inputs, desiredOutputs)
# b = newModel.predict(np.expand_dims(data[0],axis = 0))