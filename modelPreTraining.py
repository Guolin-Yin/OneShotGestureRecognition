import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from Preprocess.gestureDataLoader import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from MODEL import models
from t_SNE import *
import os
import re
class PreTrainModel:
    def __init__( self,config):
        self.config = config
        self.modelObj = models( )
    def _splitData( self,x_all, y_all, source: str = '4user' ):
        '''
        This function build for split sign data, 1 to 125 for training, 125 to 150 for one shot learning
        :param x_all:
        :param y_all:
        :param source:
        :return:
        '''
        if source == '4user':
            train_data = np.zeros( (5000, 200, 60, 3) )
            train_labels = np.zeros( (5000, 1), dtype = int )
            unseen_sign_data = np.zeros( (1000, 200, 60, 3) )
            unseen_sign_label = np.zeros( (1000, 1), dtype = int )
            count_tra = 0
            count_test = 0
            for i in np.arange( 0, 6000, 1500 ):
                train_data[ count_tra:count_tra + 1250, :, :, : ] = x_all[ i:i + 1250, :, :, : ]
                train_labels[ count_tra:count_tra + 1250, : ] = y_all[ i:i + 1250, : ]
                unseen_sign_data[ count_test:count_test + 250, :, :, : ] = x_all[ i + 1250:i + 1500, :, :, : ]
                unseen_sign_label[ count_test:count_test + 250, : ] = y_all[ i + 1250:i + 1500, : ]
                count_tra += 1250
                count_test += 250
            idx = np.random.permutation( len( train_labels ) )
            return [ train_data[ idx ], train_labels[ idx ], unseen_sign_data, unseen_sign_label ]
        elif source == 'singleuser':
            x_all = x_all[ 0:1250 ]
            y_all = y_all[ 0:1250 ]
            idx = np.random.permutation( len( y_all ) )
            x_all = x_all[ idx, :, :, : ]
            y_all = y_all[ idx, : ]
        return [ x_all, y_all ]
    def _getOneshotTaskData(self, test_data, test_labels, nway, mode:str = 'cross_val' ):
        '''
        This function build for n-way 1 shot task
        :param test_data: the Data for testing model
        :param test_labels: corresponding labels
        :param nway: the number of training classes
        :param mode: cross validation or fix the support set classes
        :return: support set : one sample, query set one sample
        '''
        signRange = np.arange( int( np.min( test_labels ) ), int( np.max( test_labels ) + 1 ), 1 )
        # signRange = np.arange( int( 251 ), int( np.max( test_labels ) + 1 ), 1 )
        selected_Sign = np.random.choice( signRange, size=nway, replace=False )
        support_set = [ ]
        query_set = [ ]
        labels = [ ]
        for i in selected_Sign:
            index, _ = np.where( test_labels == i )
            if mode == 'cross_val':
                selected_samples = np.random.choice( index, size=self.config.nshots+1, replace=False )
                n_idx = len(selected_samples)
                support_set.append( test_data[ selected_samples[ 0:n_idx-1 ] ] )
                query_set.append( test_data[ selected_samples[ -1 ] ] )
                labels.append( i )
            elif mode == 'fix':
                selected_samples = np.random.choice( index[1:], size=1, replace=False )
                support_set.append( test_data[ index[ 0 ] ] )
                query_set.append( test_data[ selected_samples[ 0 ] ] )
        return np.concatenate( support_set,axis=0 ), query_set
    def builPretrainModel( self,mode):
        '''
        This function build for create the pretrain model
        :param mode: select backbone model
        :return: whole model for pre-training and feature extractor
        '''

        if 'Alexnet' in mode :
            self.feature_extractor = self.modelObj.buildFeatureExtractor( mode = mode )
            self.feature_extractor.trainable = True
            input = Input( self.config.input_shape, name = 'data input' )
            feature_extractor = self.feature_extractor(input)
            full_connect = Dense(
                    units = self.config.N_train_classes,
                    bias_regularizer = regularizers.l2( 4e-4 ),
                    )( feature_extractor )
            output = Softmax( )( full_connect )
            preTrain_model = Model( inputs = input, outputs = output )
            # Complie preTrain_model
            # optimizer = tf.keras.optimizers.SGD(
            #         learning_rate =self.config.lr,
            #         momentum = 0.99
            #         )
            optimizer = tf.keras.optimizers.Adamax(
                    learning_rate = self.config.lr, beta_1 = 0.95, beta_2 = 0.99, epsilon = 1e-09,
                    name = 'Adamax'
                    )
            preTrain_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
            preTrain_model.summary( )
        return preTrain_model, self.feature_extractor
    def signTest(self, test_data, test_labels, N_test_sample, embedding_model, mode:str = 'cross_val' ):
        def _getNShotsEmbedding(feature_extractor,support_set):
            N_shotsEmbedding = feature_extractor.predict(np.asarray(support_set))
            cls_idx = np.arange(0,len(N_shotsEmbedding),self.config.nshots)
            embeddings_out = []
            for i in cls_idx:
                embeddings_out.append(np.mean(N_shotsEmbedding[i:i+self.config.nshots],axis = 0))
            return np.asarray(embeddings_out)
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
        nway_max = 26
        test_acc = [ ]
        softmax_func = tf.keras.layers.Softmax( )
        # for nway in np.concatenate((np.arange(2,10),np.arange(10,77,10)),axis=0):
        for nway in [70]:
            print( "Checking %d way accuracy...." % nway )
            correct_count = 0
            for i in range( N_test_sample ):
                support_set, query_set = self._getOneshotTaskData( test_data, test_labels, nway=nway, mode = mode)
                sample_index = random.randint( 0, nway - 1 )
                if mode == 'fix' and i == 0:
                    support_set_embedding = embedding_model.predict( np.asarray( support_set ) )
                elif mode == 'cross_val':
                    # support_set_embedding = embedding_model.predict( np.asarray( support_set ) )
                    support_set_embedding = _getNShotsEmbedding( embedding_model,np.asarray( support_set ) )
                query_set_embedding = embedding_model.predict( np.expand_dims( query_set[ sample_index ], axis=0 ) )
                sim = cosine_similarity( support_set_embedding, query_set_embedding )
                prob = softmax_func( np.squeeze( sim, -1 ) ).numpy()
                if np.argmax( prob ) == sample_index:
                    correct_count += 1
            acc = (correct_count / N_test_sample) * 100.
            test_acc.append( acc )
            print( "Accuracy %.2f" % acc )
        return test_acc
    # def reshapeData(self, x,mode:str = 'reshape'):
    #     if mode == 'reshape':
    #         x = x.reshape( np.shape( x )[ 0 ], x.shape[ 2 ], x.shape[ 1 ] )
    #         return x
    #     if mode == 'transpose':
    #         out = np.zeros((x.shape[0],x.shape[2],x.shape[1]))
    #         for i in range(x.shape[0]):
    #             out[i,:,:] = x[i,:,:].transpose()
    #         return out
    def scheduler(self, epoch, lr):
        if epoch > 100:
            return lr * tf.math.exp(-0.1)
        else:
            return lr
def train_user_1to5():
    config = getConfig( )
    config.source = 'lab'

    # Declare objects
    dataLoadObj = signDataLoader( dataDir=config.train_dir )
    preTrain_modelObj = PreTrainModel( )
    # Training params
    lrScheduler = tf.keras.callbacks.LearningRateScheduler( preTrain_modelObj.scheduler )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=20, restore_best_weights=True )
    # Sign recognition
    x_all, y_all,_,_ = dataLoadObj.getFormatedData(source = config.source)
    [train_data,train_labels,test_data,test_labels] = preTrain_modelObj._splitData( x_all, y_all )

    train_labels = to_categorical(train_labels - 1,num_classes=int(np.max(train_labels)))
    preTrain_model, feature_extractor = preTrain_modelObj.builPretrainModel( mode = 'Alexnet' )
    history = preTrain_model.fit(
            train_data, train_labels, validation_split = 0.1,
            epochs = 1000,
            callbacks = [ earlyStop, lrScheduler ]
            )
    val_acc = history.history[ 'val_acc' ]
    config.setSavePath( val_acc = val_acc )
    feature_extractor.save_weights(config.feature_extractor_save_path)
def train_lab(N_train_classes):
    config = getConfig( )
    config.source = 'lab'
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    config.N_train_classes = N_train_classes
    config.lr = 3e-4
    # Declare objects
    dataLoadObj = signDataLoader( dataDir = config.train_dir ,config = config,)
    preTrain_modelObj = PreTrainModel( config = config )
    # Training params
    lrScheduler = tf.keras.callbacks.LearningRateScheduler( preTrain_modelObj.scheduler )
    earlyStop = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 20, restore_best_weights = True )
    # Sign recognition
    train_data, train_labels, test_data, test_labels = dataLoadObj.getFormatedData(
            source = config.source,
            isZscore = False
            )
    train_labels = to_categorical( train_labels - 1, num_classes = int( np.max( train_labels ) ) )
    preTrain_model, feature_extractor = preTrain_modelObj.builPretrainModel( mode = 'Alexnet' )
    history = preTrain_model.fit(
            train_data, train_labels, validation_split = 0.05,
            epochs = 1000,
            callbacks = [ earlyStop, lrScheduler ]
            )
    val_acc = history.history[ 'val_acc' ]
    return [preTrain_model, feature_extractor,val_acc,config]
def test(FE_path,mode,N_train_classes,environment:str):
    config = getConfig( )
    modelObj = models( )
    config.source = environment
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    config.N_train_classes = N_train_classes
    config.nshots = 5
    # config.lr = 3e-4
    config.pretrainedfeatureExtractor_path = FE_path
    # Declare objects
    dataLoadObj = signDataLoader( config = config )
    preTrain_modelObj = PreTrainModel( config = config )
    train_data, train_labels, test_data, test_labels = dataLoadObj.getFormatedData(
            source = config.source,
            isZscore = False
            )
    feature_extractor = modelObj.buildFeatureExtractor( mode = mode )
    feature_extractor.load_weights(config.pretrainedfeatureExtractor_path )
    test_acc = preTrain_modelObj.signTest( test_data, test_labels, 100, feature_extractor )

    # predict_lab = feature_extractor.predict( test_data )
    # WidarDataloaderObj = WidarDataloader( 'E:/Cross_dataset/20181109/User1', selection = (2, 2, 3) )
    # data_widar = WidarDataloaderObj.getSQDataForTest( 1, mode = 'fix' )[ 'Val_data' ]
    # label_widar = WidarDataloaderObj.getSQDataForTest( 1, mode = 'fix' )[ 'Val_label' ]
    # predict_widar =  feature_extractor.predict(data_widar)
    # domain_t_sne( ( predict_lab,predict_home,predict_widar), perplexity = 7, n_iter = 2000 )
    # class_t_sne(predict_widar.reshape(len(predict_widar),-1),label_widar, perplexity = 7, n_iter = 2000)
    # label_range = [ 251, 261, 256, 270, 271 ]
    # idx = np.where( test_labels == label_range )[ 0 ]
    return test_acc
if __name__ == '__main__':
    # preTrain_model, feature_extractor = train_lab()
    # config = getConfig( )
    # config.pretrainedfeatureExtractor_path = \
    #     f'./models/feature_extractor_weight_Alexnet_lab_250cls_val_acc_0.996_no_zscore.h5'
    # feature_extractor.save_weights( config.pretrainedfeatureExtractor_path )
    # test_acc = test('./models/Using_CSI_ratio_model/feature_extractor_weight_Alexnet_lab_250cls_val_acc_0.92_CSIRatio'
    #                 '.h5',mode = "Alexnet3")

    # test_acc_2 = test( './models/signFi_featureExtractor_weight_AlexNet_lab_training_acc_0.95_on_250cls.h5',
    #         mode = "Alexnet" )

    '''Training'''
    # [preTrain_model, feature_extractor,val_acc,config] = train_lab(N_train_classes)
    # extractor_path = f'./models/signFi_featureExtractor_weight_AlexNet_lab_training_acc_{np.max(val_acc):.2f}_on' \
    #                   f'_{config.N_train_classes}cls.h5'
    # feature_extractor.save_weights( extractor_path )
    '''Testing'''
    envirs = ['home','lab']
    all_acc = { }
    all_path = os.listdir( f'./models/pretrained_feature_extractors/' )
    for environment in envirs:
        for i,path in enumerate(all_path):
            n = re.findall( r'\d+', all_path[ i ] )[2]
            if int(n) == 200:
                print( f'{n} in environment {environment}' )
                extractor_path = './models/pretrained_feature_extractors/' + path
                acc = test(
                        FE_path = extractor_path, mode = 'Alexnet', N_train_classes = int( n ),
                        environment = environment
                        )
                all_acc[f'{n}_{environment}'] = np.asarray(acc)
    # savemat('./models/result.mat',all_acc)