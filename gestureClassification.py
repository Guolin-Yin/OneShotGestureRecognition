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
from saveData import preprocessData
'''Initialization parameters'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
config = getConfig()
def defineModel():
    embedding = SiamesNetworkTriplet_2(batch_size=32,lr = 0.001)
    network = embedding.build_embedding_network()
    input = Input([1600,7],name='data input')
    encoded_model = network(input)
    output = Dense(units = config.num_classes, activation= 'softmax')(encoded_model)
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
def Testing( test_dir:str,embedding_model,N_test_sample:int,isOneShotTask:bool=True ):
    nway_min = 2
    nway_max = 6
    test_acc = [ ]
    def averageSim(x):
        return np.mean(x)
    # nway_list = [ ]
    for nway in range( nway_min, nway_max + 1 ):
        print( "Checking %d way accuracy...." % nway )
        correct_count = 0
        if nway == 1:
            for _ in range( N_test_sample ):
                threshold =0
                nway_anchor, nway_positive, _ = gestureDataLoader( data_path=test_dir,
                                                                   batch_size=nway ).getTripletTrainBatcher( )
                nway_anchor = reshapeData(nway_anchor)
                nway_positive = reshapeData(nway_positive)
                nway_anchor_embedding = embedding_model.predict( nway_anchor )
                sample_index = random.randint( 0, nway - 1 )
                sample_embedding = embedding_model.predict( np.expand_dims( nway_positive[ sample_index ], axis=0 ) )
                sim = cosine_similarity( nway_anchor_embedding, sample_embedding )
                if sim >= threshold:
                    correct_count += 1
            acc = (correct_count / N_test_sample) * 100.
            print( "Accuracy %.2f" % acc )
        if nway > 1:
            if isOneShotTask:
                for _ in range( N_test_sample ):
                    # Retrieving nway number of triplets and calculating embedding vector
                    nway_anchor, nway_positive,_ = gestureDataLoader( data_path=test_dir,
                                                                      batch_size=nway ).getTripletTrainBatcher( isOneShotTask=isOneShotTask )
                    nway_anchor = np.asarray( list(map(preprocessData,nway_anchor)))
                    nway_positive = np.asarray(list(map(preprocessData,nway_positive)))

                    nway_positive = reshapeData( nway_positive)
                    nway_anchor = reshapeData( nway_anchor)
                    # support set, it has N different classes depending on the batch_size
                    # nway_anchor has the same class with nway_positive at the same row
                    sample_index = random.randint( 0, nway - 1 )
                    nway_anchor_embedding = embedding_model.predict( nway_anchor )
                    sample_embedding = embedding_model.predict(np.expand_dims( nway_positive[ sample_index ], axis=0 ) )
                    # using cosine_similarity
                    sim = cosine_similarity( nway_anchor_embedding, sample_embedding )
                    if np.argmax( sim ) == sample_index:
                        correct_count += 1
                #   nway_list.append( nway )
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
            if not isOneShotTask:
                for _ in range( N_test_sample ):
                    sim = []
                    # Retrieving nway number of triplets and calculating embedding vector
                    nway_anchor, nway_positive= gestureDataLoader( data_path=test_dir,
                                                                   batch_size=nway ).getTripletTrainBatcher(
                        isOneShotTask=False, nShots=20 )

                    nway_positive = reshapeData(nway_positive)
                    # support set, it has N different classes depending on the batch_size
                    # nway_anchor has the same class with nway_positive at the same row

                    sample_index = random.randint( 0, nway - 1 )
                    sample_embedding = embedding_model.predict( np.expand_dims( nway_positive[ sample_index ], axis=0 ) )

                    for nB in range(nway):
                        nway_anchor_nB = reshapeData( nway_anchor[nB] )
                        nway_anchor_embedding_for_batch = embedding_model.predict( nway_anchor_nB )
                        sim_nb_batch = averageSim(cosine_similarity( nway_anchor_embedding_for_batch, sample_embedding ))
                        sim.append(sim_nb_batch)
                    if np.argmax( sim ) == sample_index:
                        correct_count += 1
            #   nway_list.append( nway )
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
# load data
def loadData(dataDir):
    print('Loading data.....................................')

    data = []
    labels = []
    gesture_6 = ['E:/Widar_dataset_matfiles/20181109/User1',
                'E:/Widar_dataset_matfiles/20181109/User2',]
    gesture_10 = ['E:/Widar_dataset_matfiles/20181112/User1',
                  'E:/Widar_dataset_matfiles/20181112/User2',
                  'Combined_link_dataset/20181116']
    for Dir in dataDir:
        fileName = os.listdir( Dir )
        for name in fileName:
            if re.findall( r'\d+\b', name )[5] == '3':
                print(f'Loading {name}')
                path = os.path.join( Dir, name )
                data.append(preprocessData(sio.loadmat(path)['csiAmplitude']))
                if Dir in gesture_6:
                    gestureMark = int(re.findall( r'\d+\b', name )[ 1 ]) - 1
                elif Dir in gesture_10:
                    gestureMark = int( re.findall( r'\d+\b', name )[ 1 ] ) + 6 - 1
                labels.append(tf.keras.utils.to_categorical(gestureMark,num_classes=config.num_classes))
    return np.asarray(data),np.asarray(labels)
def reshapeData(x,mode:str = 'reshape'):
    if mode == 'reshape':
        x = x.reshape( np.shape( x )[ 0 ], x.shape[ 2 ], x.shape[ 1 ] )
        return x
    if mode == 'transpose':
        out = np.zeros((x.shape[0],x.shape[2],x.shape[1]))
        for i in range(x.shape[0]):
            out[i,:,:] = x[i,:,:].transpose()
        return out

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.2)
if __name__ == '__main__':

    data,labels = loadData(dataDir = config.train_dir)
    X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.5,shuffle = True)
    X_train = reshapeData(X_train,mode = 'reshape')
    model,network = defineModel()

    lrScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        batch_size=32, epochs=50,
                        callbacks = [lrScheduler,earlyStop])
    model.evaluate(reshapeData(X_test,mode='reshape'), y_test)


    # saving the weights for trained
    network.save_weights( './models/similarity_featureExtractor_weights_task2_single_link_16class_half_samples.h5' )
    # model.save_weights('./models/similarity_whole_model_weights_task3_six_link.h5')
# Output for sipecific layer
# desiredLayers = [15]
# desiredOutputs = [network.layers[i].output for i in desiredLayers]
# newModel = Model(network.inputs, desiredOutputs)
# b = newModel.predict(np.expand_dims(data[0],axis = 0))