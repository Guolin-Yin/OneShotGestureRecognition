import tensorflow as tf
import numpy as np
from modelPreTraining import *
from Preprocess.gestureDataLoader import signDataLoder
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, Lambda,Dot,concatenate,ZeroPadding2D,Conv2D,\
    MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from MODEL import models
import random
class fineTuningModel:
    def __init__( self,nshots,isZscore,config ):
        self.nshots = nshots
        self.isZscore = isZscore
        self.modelObj = models( )
        self.config = config
        self.trainTestObj = PreTrainModel(config = config )
        self.lrScheduler = tf.keras.callbacks.LearningRateScheduler( self.trainTestObj.scheduler )
        self.earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights
        =True)
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        self.pretrained_featureExtractor.trainable = True
        self.data = self._getSQData( nshots = nshots)
        if nshots == 1:
            self.isOneShotTask = True
        else:
            self.isOneShotTask = False
    def _getSQData( self,nshots:int ):
        '''
        This function build for split support set and query set according to the number of shots
        :param nshots:
        :return:
        '''
        testSign = signDataLoder( dataDir = config.train_dir )
        fixed_query_set = False
        if type(config.source) == list or 'user' in config.source:
            _, _, x_test, y_test = testSign.getFormatedData( source = config.source,isZscore = self.isZscore )
            x = x_test[ 1250:1500 ]
            y = y_test[ 1250:1500 ]
            num = nshots * 25
            Support_data = x[ 0:num, :, :, : ]
            Support_label = y[0:num,:]
            Query_data = x[num:len(x)+1,:,:,:]
            Query_label = y[num:len(x)+1,:]
        elif 'home' in config.source or 'lab' in config.source:
            if fixed_query_set:
                _, _, x_test, y_test = testSign.getFormatedData( source = config.source, isZscore = self.isZscore )
                num = nshots * config.num_finetune_classes
                query_idx = 5 * config.num_finetune_classes
                Support_data = x_test[ 0:num, :, :, : ]
                Support_label = y_test[ 0:num, : ]
                Query_data = x_test[ query_idx:len( x_test ) + 1, :, :, : ]
                Query_label = y_test[ query_idx:len( x_test ) + 1, : ]
            else:
                _, _, x_test, y_test = testSign.getFormatedData( source = config.source,isZscore=self.isZscore )
                num = nshots * 26
                Support_data = x_test[ 0:num, :, :, : ]
                Support_label = y_test[ 0:num, : ]
                Query_data = x_test[ num:len( x_test ) + 1, :, :, : ]
                Query_label = y_test[ num:len( x_test ) + 1, : ]
        output = {'Support_data':Support_data,
                  'Support_label':Support_label,
                  'Query_data':Query_data,
                  'Query_label':Query_label}
        return output
    def _getValData( self ):
        '''
        Get the validation data for fine tuning
        :return:
        '''
        val_data = self.data['Query_data']
        val_label = to_categorical(
                self.data[ 'Query_label' ] - np.min( self.data[ 'Query_label' ] ), num_classes =
                config.num_finetune_classes
                )
        return [val_data,val_label]
    def _getDataToTesting(self,query_set,nway,mode:str = 'fix'):
        '''
        Randomly choose one sample from the Query data set
        :param query_set:
        :param nway:
        :return:
        '''
        if mode == 'fix':
            sample_sign = np.random.choice(np.arange(0,len(query_set),config.num_finetune_classes),size = 1,replace = False)
            sample_index = random.randint( 0, nway - 1 )
            query_data = np.repeat( query_set[ sample_sign+sample_index ], [ nway ], axis = 0 )
            return [ query_data, sample_index ]
        elif mode == 'random':
            sample_sign = np.random.choice(
                    np.arange( 0, len( query_set ), config.num_finetune_classes ), size = 2, replace = False
                    )
            sample_index = random.randint( 0, nway - 1 )
            support_data = np.repeat( query_set[ sample_sign[0] + sample_index ], [ nway ], axis = 0 )
            query_data = np.repeat( query_set[ sample_sign[1] + sample_index ], [ nway ], axis = 0 )
            return [ support_data,query_data, sample_index ]

    def _getNShotsEmbedding( self,featureExtractor, Support_data):
        Sign_class = np.arange( 0, self.config.num_finetune_classes, 1 )
        # Sign_samples = np.arange( 0, 125, 25 )
        Sign_samples = np.arange( 0, len(Support_data), self.config.num_finetune_classes )
        n_shots_support_embedding = [ ]
        for i in Sign_class:
            n_shots_support_data = [ ]
            for j in Sign_samples:
                n_shots_support_data.append( Support_data[ i + j ] )
            n_shots_support_embedding.append(np.mean( featureExtractor.predict( np.asarray( n_shots_support_data ) ), axis = 0 ) )
        n_shots_support_embedding = np.asarray( n_shots_support_embedding )
        return n_shots_support_embedding
    def _getPreTrainedFeatureExtractor( self ):
        '''
        This function build for recreating the feature extractor and load pre-trained weights
        :return: feature extractor
        '''
        trained_featureExtractor = self.modelObj.buildFeatureExtractor( mode='Alexnet' )
        trained_featureExtractor.load_weights(config.pretrainedfeatureExtractor_path )
        return trained_featureExtractor
    def _loadFineTunedModel(self,applyFinetunedModel:bool = True):
        '''
        This function build for load fine tuned model for testing
        :returns pre-trained feature extractor and fine tuned classifier
        '''

        if applyFinetunedModel:
            print( f'loading fine tuned model: {config.tunedModel_path}' )
            fine_Tune_model = self.modelObj.buildTuneModel( config = self.config,isTest = True )
            fine_Tune_model.load_weights(config.tunedModel_path)
            feature_extractor = Model(
                    inputs = fine_Tune_model.input, outputs = fine_Tune_model.get_layer( 'fine_tune_layer' ).output
                    )
        elif not applyFinetunedModel:
            print( f'loading original pretrained feature extractor: {config.pretrainedfeatureExtractor_path}' )
            feature_extractor = self.pretrained_featureExtractor
        '''
        Classifier input: two feature vector
                      output: one probability
        '''
        cls_intput_Support = Input(feature_extractor.output.shape[1],name = 'Support_input')
        cls_intput_Query = Input( feature_extractor.output.shape[1], name = 'Query_input' )
        cosSim_layer = Dot( axes = 1, normalize = True )([cls_intput_Support,cls_intput_Query])
        cls_output = Softmax( )( tf.squeeze(cosSim_layer,-1) )
        classifier = Model(inputs = [cls_intput_Support,cls_intput_Query],outputs = cls_output)
        # feature_extractor, classifier = self._configModel(model = self.fine_Tune_model)
        return [feature_extractor, classifier]
    def tuning( self ,init_weights = True,init_bias = False):
        fine_Tune_model = self.modelObj.buildTuneModel(
                pretrained_feature_extractor = self.pretrained_featureExtractor,
                isTest = False,config = self.config
                )
        if init_weights:
            if self.nshots == 1:
                weights = np.transpose( self.pretrained_featureExtractor.predict( self.data[ 'Support_data' ] ) )
            elif self.nshots > 1:
                weights = np.transpose(self._getNShotsEmbedding( self.pretrained_featureExtractor,self.data[ 'Support_data' ] ) )
            if init_bias:
                p = fine_Tune_model.predict(self.data['Query_data'])
                bias = np.tile(np.mean(-np.sum( p * np.log(p ),axis = 1 ) ),config.num_finetune_classes)
            else:
                bias = np.zeros(config.num_finetune_classes)
            fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )

        val_data, val_label = self._getValData( )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = config.lr,
                # momentum = 0.9,
                epsilon = 1e-07,
                )
        # optimizer = tf.keras.optimizers.SGD(
        #         learning_rate = config.lr,
        #         momentum = 0.9,
        #         )
        fine_Tune_model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
        idx = np.random.permutation(len(self.data[ 'Support_data' ]))
        fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical(self.data[ 'Support_label' ][ idx ] - np.min(
                                self.data[ 'Support_label' ]),num_classes = config.num_finetune_classes),
                epochs = 1000,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return fine_Tune_model
    def test( self, applyFinetunedModel:bool=True ):
        nway = config.num_finetune_classes
        N_test_sample = 1000
        correct_count = 0
        test_acc = []
        # load Support and Query dataset
        query_set, query_label = self._getValData( )
        Support_data = self.data[ 'Support_data' ]
        feature_extractor, classifier = self._loadFineTunedModel( applyFinetunedModel )
        if self.isOneShotTask:
            Support_set_embedding = feature_extractor.predict( Support_data )
            for i in range(N_test_sample):
                Query_data, sample_index = self._getDataToTesting( query_set = query_set, nway = nway )
                Query_set_embedding = feature_extractor.predict( Query_data )
                prob_classifier = classifier.predict([Support_set_embedding,Query_set_embedding])
                if np.argmax( prob_classifier ) == sample_index:
                    correct_count += 1
                    print( f'The number of correct: {correct_count}, The number of test count {i}' )
            acc = (correct_count / N_test_sample) * 100.
            test_acc.append( acc )
            print( "Accuracy %.2f" % acc )
            # Test method 2:
            # optimizer = tf.keras.optimizers.SGD( learning_rate = config.lr, momentum = 0.9 )
            # self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
            # self.fine_Tune_model.evaluate( query_set, query_label )
        if not self.isOneShotTask:
            Support_set_embedding = self._getNShotsEmbedding( feature_extractor,Support_data )
            for i in range(N_test_sample):
                Query_data, sample_index = self._getDataToTesting( query_set = query_set, nway = nway )
                Query_set_embedding = feature_extractor.predict( Query_data )
                prob_classifier = classifier.predict( [ Support_set_embedding, Query_set_embedding ] )
                if np.argmax( prob_classifier ) == sample_index:
                    correct_count += 1
                    print( f'The number of correct: {correct_count}, The number of test count {i}' )
            acc = (correct_count / N_test_sample) * 100.
            test_acc.append( acc )
            print( "Accuracy %.2f" % acc )
        return test_acc

if __name__ == '__main__':
    # print('start')
    config = getConfig( )
    config.source = 'lab'
    config.num_finetune_classes = 26
    config.pretrainedfeatureExtractor_path = \
        './models/feature_extractor_weight_Alexnet_home_250cls_val_acc_0.97_with_Zscore.h5'
    'signFi_featureExtractor_weight_AlexNet_lab_training_acc_0.95_on_250cls.h5'
    nshots = 10

    fineTuningModelObj = fineTuningModel(nshots = nshots,isZscore = True,config = config)
    # Tuning
    fine_Tune_model = fineTuningModelObj.tuning(init_bias= False)
    path = f'./models/fc_fineTuned_250Cls_homeTolab_{nshots}_shot_with_Zscore.h5'
    fine_Tune_model.save_weights(path)
    # Testing
    config.tunedModel_path = path
    test_acc = fineTuningModelObj.test(applyFinetunedModel = True)
    print('Done')






