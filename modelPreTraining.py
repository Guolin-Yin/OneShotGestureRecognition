import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Softmax
from tensorflow.keras.models import Model
from Preprocess.gestureDataLoader import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from MODEL import models
config = getConfig()
class PreTrainModel:
    def __init__( self,mode:str = 'Alexnet' ):
        modelObj = models( )
        self.feature_extractor = modelObj.buildFeatureExtractor( mode = mode )
    def builPretrainModel( self,mode: str = '1D' ):
        '''
        This function build for create the pretrain model
        :param mode: select backbone model
        :return: whole model for pre-training and feature extractor
        '''
        embedding = models()
        network = embedding.buildFeatureExtractor( mode='Alexnet' )
        if mode == '1D':
            input = Input( [ 1600, 7 ], name = 'data input' )
            feature_extractor = network( input )
            output = Dense( units = config.N_train_classes, activation = 'softmax' )( feature_extractor )
            preTrain_model = Model( inputs = input, outputs = output )
            optimizer = tf.keras.optimizers.Adam(
                    lr = 0.001,
                    beta_1 = 0.9,
                    beta_2 = 0.999,
                    epsilon = 1e-07,
                    amsgrad = False,
                    # lr_multipliers=learning_rate_multipliers,
                    )
            preTrain_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
            preTrain_model.summary( )
        elif mode == '2D':
            # Define preTrain_model
            input = Input( config.input_shape, name = 'data input' )
            feature_extractor = network( input )
            full_connect = Dense( units = config.N_train_classes )( feature_extractor )
            output = Softmax( )( full_connect )
            preTrain_model = Model( inputs = input, outputs = output )
            # Complie preTrain_model
            optimizer = tf.keras.optimizers.SGD(
                    learning_rate = config.lr, momentum = 0.9
                    )
            preTrain_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
            preTrain_model.summary( )
        elif mode == 'Alexnet':
            input = Input( config.input_shape, name = 'data input' )
            feature_extractor = self.feature_extractor( input )
            full_connect = Dense( units = config.N_train_classes )( feature_extractor )
            output = Softmax( )( full_connect )
            preTrain_model = Model( inputs = input, outputs = output )
            # Complie preTrain_model
            optimizer = tf.keras.optimizers.SGD(
                    learning_rate = config.lr, momentum = 0.9
                    )
            preTrain_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
            preTrain_model.summary( )
        return preTrain_model, self.feature_extractor
    def labTrainData( self,x_all, y_all, source: str = 'user1to4' ):
        if source == 'user1to4':
            train_data = np.zeros( (5000, 200, 60, 3) )
            train_labels = np.zeros( (5000, 1), dtype = int )
            test_data = np.zeros( (1000, 200, 60, 3) )
            test_labels = np.zeros( (1000, 1), dtype = int )
            count_tra = 0
            count_test = 0
            for i in np.arange( 0, 6000, 1500 ):
                train_data[ count_tra:count_tra + 1250, :, :, : ] = x_all[ i:i + 1250, :, :, : ]
                train_labels[ count_tra:count_tra + 1250, : ] = y_all[ i:i + 1250, : ]
                test_data[ count_test:count_test + 250, :, :, : ] = x_all[ i + 1250:i + 1500, :, :, : ]
                test_labels[ count_test:count_test + 250, : ] = y_all[ i + 1250:i + 1500, : ]
                count_tra += 1250
                count_test += 250
            idx = np.random.permutation( len( train_labels ) )
            return [ train_data[ idx ], train_labels[ idx ], test_data, test_labels ]
        elif source == 'singleuser':
            x_all = x_all[ 0:1250 ]
            y_all = y_all[ 0:1250 ]
            idx = np.random.permutation( len( y_all ) )
            x_all = x_all[ idx, :, :, : ]
            y_all = y_all[ idx, : ]
        return [ x_all, y_all ]
    def _getOneshotTaskData(self, test_data, test_labels, nway,mode:str = 'cross_val' ):
        '''
        This function build for n-way 1 shot task
        :param test_data: the Data for testing model
        :param test_labels: corresponding labels
        :param nway: the number of training classes
        :param mode: 1. cross validation or fix the support set classes
        :return: support set : one sample, query set one sample
        '''
        signRange = np.arange( int( np.min( test_labels ) ), int( np.max( test_labels ) + 1 ), 1 )
        selected_Sign = np.random.choice( signRange, size=nway, replace=False )
        support_set = [ ]
        query_set = [ ]
        for i in selected_Sign:
            index, _ = np.where( test_labels == i )
            if mode == 'cross_val':
                selected_samples = np.random.choice( index, size=2, replace=False )
                support_set.append( test_data[ selected_samples[ 0 ] ] )
                query_set.append( test_data[ selected_samples[ 1 ] ] )
            elif mode == 'fix':
                selected_samples = np.random.choice( index[1:], size=1, replace=False )
                support_set.append( test_data[ index[ 0 ] ] )
                query_set.append( test_data[ selected_samples[ 0 ] ] )
        return support_set, query_set
    def Test( self, test_data, test_labels, mode:str = 'fix',isOneShotTask: bool = True):
        '''
        This method build for testing fix number of ways
        :param test_data:
        :param test_labels:
        :param mode:
        :param isOneShotTask:
        :return:
        '''
        nway = 5
        N_test_sample = 1000
        test_acc = [ ]
        softmax_func = tf.keras.layers.Softmax( )
        for _ in range( N_test_sample ):
            if isOneShotTask:
                support_set, query_set, _ = self._getOneshotTaskData( test_data, test_labels, nway=nway, mode=mode )
                sample_index = random.randint( 0, nway - 1 )
                query_sample = np.repeat(query_set[sample_index],[nway],axis = 0)
                sim = self.embedding_model.predict( [ support_set, query_sample ] )
                prob = softmax_func( np.squeeze( sim, -1 ) ).numpy( )
                if np.argmax( prob ) == sample_index:
                    correct_count += 1
        acc = (correct_count / N_test_sample) * 100.
        test_acc.append( acc )
        print( "Accuracy %.2f" % acc )
    def signTest(self, test_data, test_labels, N_test_sample, embedding_model, isOneShotTask: bool = True, mode:str = 'cross_val' ):
        '''
        This function build for testing the model performance from two ways to 25 ways
        :param test_data:
        :param test_labels:
        :param N_test_sample:
        :param embedding_model:
        :param isOneShotTask:
        :param mode:
        :return:
        '''
        nway_min = 2
        nway_max = 25
        test_acc = [ ]
        softmax_func = tf.keras.layers.Softmax( )
        for nway in range( nway_min, nway_max + 1 ):
            print( "Checking %d way accuracy...." % nway )
            correct_count = 0
            if isOneShotTask:
                for _ in range( N_test_sample ):
                    # Retrieving nway number of triplets and calculating embedding vector
                    support_set, query_set,_ = self._getOneshotTaskData( test_data, test_labels, nway=nway, mode = mode)
                    # support set, it has N different classes depending on the batch_size
                    # nway_anchor has the same class with nway_positive at the same row
                    sample_index = random.randint( 0, nway - 1 )
                    support_set_embedding = embedding_model.predict( np.asarray( support_set ) )
                    # support_set_embedding = normalize( support_set_embedding, axis=1, norm='max' )
                    query_set_embedding = embedding_model.predict( np.expand_dims( query_set[ sample_index ], axis=0 ) )
                    # query_set_embedding = normalize( query_set_embedding, axis=1, norm='max' )
                    # using cosine_similarity
                    sim = cosine_similarity( support_set_embedding, query_set_embedding )
                    prob = softmax_func( np.squeeze( sim, -1 ) ).numpy()
                    if np.argmax( prob ) == sample_index:
                        correct_count += 1
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
        return test_acc
    def reshapeData(self, x,mode:str = 'reshape'):
        if mode == 'reshape':
            x = x.reshape( np.shape( x )[ 0 ], x.shape[ 2 ], x.shape[ 1 ] )
            return x
        if mode == 'transpose':
            out = np.zeros((x.shape[0],x.shape[2],x.shape[1]))
            for i in range(x.shape[0]):
                out[i,:,:] = x[i,:,:].transpose()
            return out
    def scheduler(self, epoch, lr):
        if epoch < 100:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

if __name__ == '__main__':
    # Declare objects
    dataLoadObj = signDataLoder( dataDir=config.train_dir )
    trainTestObj = PreTrainModel( )
    # Training params
    lrScheduler = tf.keras.callbacks.LearningRateScheduler( trainTestObj.scheduler )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=20, restore_best_weights=True )
    # Sign recognition
    x_all, y_all = dataLoadObj.getFormatedData(source = 'user1to4' )
    [train_data,train_labels,test_data,test_labels] = trainTestObj.labTrainData(x_all, y_all)

    train_labels = to_categorical(train_labels - 1,num_classes=int(np.max(train_labels)))
    preTrain_model, feature_extractor = trainTestObj.builPretrainModel( mode = 'Alexnet' )
    history = preTrain_model.fit(
            train_data, train_labels, validation_split = 0.2,
            epochs = 1000, shuffle = True,
            callbacks = [ earlyStop, lrScheduler ]
            )
    val_acc = history.history[ 'val_acc' ]
    save_path = f'./models/signFi_wholeModel_weight_AlexNet_training_acc_{val_acc[-1]:.2f}_on_{config.N_train_classes}cls_user1to4.h5'
    feature_extractor.save_weights(save_path)