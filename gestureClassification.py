import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv1D, Conv2D, Flatten,\
    Dense,Dropout, Input, Lambda,MaxPooling2D,\
    concatenate,BatchNormalization,MaxPooling1D,Softmax
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import random
from Preprocess.gestureDataLoader import *
import numpy as np
import os
import scipy.io as sio
import re
from SiameseNetworkWithTripletLoss import SiamesWithTriplet
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from saveData import preprocessData
from sklearn.model_selection import KFold
'''Initialization parameters'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
config = getConfig()
def defineModel(mode:str = '1D'):
    embedding = SiamesWithTriplet( )
    network = embedding.build_embedding_network( mode=mode )

    if mode == '1D':
        input = Input([1600,7],name='data input')
        encoded_model = network( input )
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
    elif mode == '2D':
        # Define model
        input = Input( [ 200,60,3 ], name='data input' )
        encoded_model = network( input )
        full_connect = Dense( units=config.num_classes)( encoded_model )
        output = Softmax( )(full_connect)
        model = Model(inputs = input,outputs = output)
        # Complie model
        optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.01, momentum=0.9
        )
        model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
        model.summary( )
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
def getOneshotTaskData(test_data,test_labels,nway):
    signRange = np.arange( int(np.min( test_labels )), int(np.max( test_labels ) + 1), 1 )
    selected_Sign = np.random.choice(signRange,size=nway,replace = False)
    support_set = []
    query_set = []
    for i in selected_Sign:
        index,_ = np.where(test_labels == i)
        selected_samples = np.random.choice( index, size=2, replace=False )
        support_set.append(test_data[selected_samples[0]])
        query_set.append(test_data[selected_samples[1]])
    return support_set,query_set
def signTest(test_data,test_labels,N_test_sample,embedding_model,isOneShotTask:bool = True):
    nway_min = 2
    nway_max = 6
    test_acc = [ ]
    for nway in range( nway_min, nway_max + 1 ):
        print( "Checking %d way accuracy...." % nway )
        correct_count = 0
        if isOneShotTask:
            for _ in range( N_test_sample ):
                # Retrieving nway number of triplets and calculating embedding vector
                support_set, query_set = getOneshotTaskData( test_data, test_labels, nway=nway )
                # support set, it has N different classes depending on the batch_size
                # nway_anchor has the same class with nway_positive at the same row
                sample_index = random.randint( 0, nway - 1 )
                nway_anchor_embedding = embedding_model.predict( np.asarray( support_set) )
                sample_embedding = embedding_model.predict( np.expand_dims( query_set[ sample_index ], axis=0 ) )
                # using cosine_similarity
                sim = cosine_similarity( nway_anchor_embedding, sample_embedding )
                if np.argmax( sim ) == sample_index:
                    correct_count += 1
            #   nway_list.append( nway )
            acc = (correct_count / N_test_sample) * 100.
            test_acc.append( acc )
            print( "Accuracy %.2f" % acc )
# load data
def getTrainTestSplit(x_all,label_lab):
    train_data = np.zeros((5200,200,60,3))
    train_labels = np.zeros((5200,1))
    test_data = np.zeros((320,200,60,3))
    test_labels = np.zeros((320,1))
    count_tra = 0
    count_tes = 0
    for i in list(np.arange(0,5520,276)):
        # train_data.append(x_all[i:i+260,:,:,:])
        # train_labels.append(label_lab[i:i+260,:].reshape(-1))
        # test_data.append(x_all[i+260:i+276,:,:,:])
        # test_labels.append(label_lab[i+260:i+276].reshape(-1))
        train_data[ count_tra:count_tra + 260, :, :, : ] = x_all[ i:i + 260, :, :, : ]
        train_labels[ count_tra:count_tra + 260, : ] = label_lab[ i:i + 260, : ]
        test_data[count_tes:count_tes+16,:,:,:] = x_all[i+260:i+276,:,:,:]
        test_labels[count_tes:count_tes+16,:] = label_lab[i+260:i+276,:]
        count_tra += 260
        count_tes += 16
    return [train_data,train_labels,test_data,test_labels]
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
    if epoch < 320:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
        # return
if __name__ == '__main__':
    # Sign recognition
    data,filename = signDataLoder( dataDir=config.train_dir ).loadData()
    lrScheduler = tf.keras.callbacks.LearningRateScheduler( scheduler )
    x = data[2]['csid_lab']
    x_amp = np.abs(x)
    x_phase = np.angle(x)

    x_all = np.concatenate((x_amp,x_phase),axis = 2)
    label_lab = data[2]['label_lab']
    train_data,train_labels,test_data,test_labels = getTrainTestSplit(x_all = x_all,label_lab = label_lab)

    idx = np.random.permutation( len(train_labels) )
    train_data = train_data[idx,:,:,:]
    train_labels = train_labels[idx,:]
    train_labels = to_categorical(train_labels - 1,num_classes=int(np.max(train_labels)))
    # X_train, X_test, y_train, y_test = train_test_split( x_all, y, test_size=0.2,shuffle = True)

    # Siamese = SiamesWithTriplet( )
    # model = Siamese.build_embedding_network(mode = '2D')
    model, network = defineModel( mode = '2D')
    history = model.fit(train_data,train_labels,validation_split=0.2,
                     epochs=1000,shuffle=True
                    )




    network.save_weights('./models/signFi_network_whole_network_structure_acc=0.91_160classes.h5')
    # model.save( './models/signFi_model_whole_model_structure.h5' )
    # kf = KFold( 5, shuffle=True, random_state=42 )
    # train_idx,test_idx = kf.split(x_all)
    # fold = 0
    # for train, test in kf.split( x_all ):
    #     fold += 1
    #     print( f"Fold #{fold}" )
    #
    #     x_train = x_all[ train ]
    #     y_train = y[ train ]
    #     x_test = x_all[ test ]
    #     y_test = y[ test ]
    #     history = model.fit(x_train,y_train,validation_data=(x_test,y_test),
    #                      epochs=100,shuffle=True
    #                     # callbacks = [lrScheduler]
    #                     )
# Gesture recognition
    # data,labels = gestureDataLoader.DirectLoadData(dataDir = config.train_dir)
    # X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.5,shuffle = True)
    # X_train = reshapeData(X_train,mode = 'reshape')
    # model,network = defineModel()
    #
    # lrScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
    # history = model.fit(X_train, y_train,
    #                     validation_split=0.2,
    #                     batch_size=32, epochs=50,
    #                     callbacks = [lrScheduler,earlyStop])
    # model.evaluate(reshapeData(X_test,mode='reshape'), y_test)
    #
    #
    # # saving the weights for trained
    # network.save_weights( './models/similarity_featureExtractor_weights_task2_single_link_16class_half_samples.h5' )
    # model.save_weights('./models/similarity_whole_model_weights_task3_six_link.h5')
# Output for sipecific layer
# desiredLayers = [15]
# desiredOutputs = [network.layers[i].output for i in desiredLayers]
# newModel = Model(network.inputs, desiredOutputs)
# b = newModel.predict(np.expand_dims(data[0],axis = 0))