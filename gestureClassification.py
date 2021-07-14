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
from Config import getConfig
'''Initialization parameters'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
config = getConfig()
def defineModel(dataDir):
    embedding = SiamesNetworkTriplet_2(batch_size=32,data_dir=dataDir,lr = 0.001)
    network = embedding.build_embedding_network()
    # network.add( Lambda( lambda x: K.l2_normalize( x, axis=-1 ) ) )
    # input = Input(embedding.input_shape,name='data input')
    input = Input([1600,7],name='data input')
    encoded_model = network(input)
    dense_1 = Dense(units = 128,activation='relu')(encoded_model)
    dropOut_1 = Dropout( 0.4 )(dense_1)
    dense_2 = Dense( units=256, activation='relu' )( dropOut_1 )
    dropOut_2 = Dropout( 0.6 )(dense_2)
    dense_3 = Dense( units=128, activation='relu' )( dropOut_2 )
    output = Dense(units = 10, activation= 'softmax')(dense_3)
    model = Model(inputs = input,outputs = output )
    optimizer = tf.keras.optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            # lr_multipliers=learning_rate_multipliers,
             )
    model.compile(loss = 'categorical_crossentropy',optimizer=optimizer,metrics = 'acc')
    model.summary()
    return model,network
def Testing( test_dir:str,embedding_model,N_test_sample:int ):
    test_sample = N_test_sample
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
            sim = cosine_similarity( nway_anchor_embedding, sample_embedding )
            if np.argmax( sim ) == sample_index:
                correct_count += 1
        nway_list.append( nway )
        acc = (correct_count / test_sample) * 100.
        test_acc.append( acc )
        print( "Accuracy %.2f" % acc )
# load data
def loadData(dataDir):
    fileName = os.listdir(dataDir)
    data = []
    labels = []
    for name in fileName:
        path = os.path.join(dataDir,name)
        data.append(sio.loadmat(path)['csiAmplitude'])
        gestureMark = int(re.findall( r'\d+\b', name )[ 1 ]) - 1
        labels.append(tf.keras.utils.to_categorical(gestureMark,num_classes=10))
    return np.asarray(data),np.asarray(labels)
def reshapeData(x):
    x = x.reshape( np.shape( x )[ 0 ], x.shape[ 2 ], x.shape[ 1 ] )
    return x
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
if __name__ == '__main__':

    data,labels = loadData(dataDir = config.train_dir)
    X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.1)
    X_train = reshapeData(X_train)
    model,network = defineModel()

    lrScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
    history = model.fit(X_train, y_train,validation_split=0.1, epochs=50,callbacks = [lrScheduler,earlyStop])
    Testing(test_dir = config.eval_dir,embedding_model = network)
    # saving the weights for trained
    network.save_weights( './models/similarity_featureExtractor_weights.h5' )
    model.save_weights('./models/similarity_whole_model_weights.h5')
# Output for sipecific layer
# desiredLayers = [15]
# desiredOutputs = [network.layers[i].output for i in desiredLayers]
# newModel = Model(network.inputs, desiredOutputs)
# b = newModel.predict(np.expand_dims(data[0],axis = 0))