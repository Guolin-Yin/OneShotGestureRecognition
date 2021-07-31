import tensorflow as tf
import numpy as np
from gestureClassification import *
from Preprocess.gestureDataLoader import signDataLoder
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, Lambda,Dot,concatenate,ZeroPadding2D,Conv2D,\
    MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from MODEL import models
class fineTuningModel:
    def __init__( self,nshots ):
        self.modelObj = models( )
        self.num_finetune_classes = 25
        self.trained_featureExtractor = self._getFeatureExtractor( )
        self.trained_featureExtractor.trainable = False
        self.classifier = []
        self.input_shape = config.input_shape
        self.trainTestObj = PreTrainModel( )
        self.fine_Tune_model = self.modelObj.buildTuneModel()
        self.lrScheduler = tf.keras.callbacks.LearningRateScheduler( self.trainTestObj.scheduler )
        self.earlyStop = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 20, restore_best_weights =
        True )
        self.data = self._getSQData( nshots = nshots)
        self.fineTuned_model_path = config.tunedModel_path
    def _getSQData( self,nshots:int = 5 ):
        '''
        This function build for split support set and query set according to the number of shot
        :param nshots:
        :return:
        '''
        testSign = signDataLoder( dataDir = config.train_dir )
        if nshots == 1:
            '''For 25 way, 1 shot'''
            x_all, y_all = testSign.getFormatedData( source = 'labUser5' )
            x = x_all[ 1250:1500 ]
            y = y_all[ 1250:1500 ]
            idx,_ = np.where(y == y[0])
            Support_data = x[idx[0]:idx[1],:,:,:]
            Support_label = y[idx[0]:idx[1],:]
            Query_data = x[idx[1]:len(x)+1,:,:,:]
            Query_label = y[idx[1]:len(x)+1,:]
        if nshots == 5:
            '''For 25 way, 5 shot'''
            x_all, y_all = testSign.getFormatedData( source='labUser5' )
            x = x_all[ 1250:1500 ]
            y = y_all[ 1250:1500 ]
            num = len(y)//2
            Support_data = x[ 0:num, :, :, : ]
            Support_label = y[0:num,:]
            Query_data = x[num:len(x)+1,:,:,:]
            Query_label = y[num:len(x)+1,:]
        output = {'Support_data':Support_data,
                  'Support_label':Support_label,
                  'Query_data':Query_data,
                  'Query_label':Query_label}
        return output
    def _getFeatureExtractor( self ):
        '''
        This function build for recreating the feature extractor and load pre-trained weights
        :return: feature extractor
        '''
        trained_featureExtractor = self.modelObj.buildFeatureExtractor( mode='Alexnet' )
        trained_featureExtractor.load_weights(config.featureExtractor_path )
        return trained_featureExtractor
    def _getFineTuneTestData(self,query_set,nway):
        '''
        Randomly choose one sample from the Query data set
        :param query_set:
        :param nway:
        :return:
        '''
        sample_sign = np.random.choice(np.arange(0,len(query_set),25),size = 1,replace = False)
        sample_index = random.randint( 0, nway - 1 )
        query_data = np.repeat( query_set[ sample_sign+sample_index ], [ nway ], axis = 0 )
        return [query_data,sample_index]
    def _getNShotsEmbedding( self, Support_data):
        Sign_class = np.arange( 0, 25, 1 )
        Sign_samples = np.arange( 0, 125, 25 )
        five_shot_support_embedding = [ ]
        for i in Sign_class:
            five_shot_support_data = [ ]
            for j in Sign_samples:
                five_shot_support_data.append( Support_data[ i + j ] )
            five_shot_support_embedding.append(
                    np.mean( self.trained_featureExtractor.predict( np.asarray( five_shot_support_data ) ), axis = 0 )
                    )
        five_shot_support_embedding = np.asarray( five_shot_support_embedding )
        return five_shot_support_embedding
    def _getValData( self ):
        '''
        Get the validation data for fine tuning
        :return:
        '''
        val_data = self.data['Query_data']
        val_label = to_categorical( self.data[ 'Query_label' ] - np.min(self.data['Query_label' ]),num_classes = 25 )
        return [val_data,val_label]
    def loadFineTunedModel(self):
        '''
        This function build for load fine tuned model for testing
        :returns pre-trained feature extractor and fine tuned classifier
        '''
        self.fine_Tune_model.load_weights(config.tunedModel_path)
        feature_extractor = Model( inputs = self.fine_Tune_model.input, outputs = self.fine_Tune_model.get_layer( 'fine_tune_layer' ).output )
        '''Classifier input: two feature vector
                      output: one probability
        '''
        cls_intput_Support = Input(25,name = 'Support_input')
        cls_intput_Query = Input( 25, name = 'Query_input' )
        cosSim_layer = Dot( axes = 1, normalize = True )([cls_intput_Support,cls_intput_Query])
        cls_output = Softmax( )( tf.squeeze(cosSim_layer,-1) )
        classifier = Model(inputs = [cls_intput_Support,cls_intput_Query],outputs = cls_output)
        # feature_extractor, classifier = self._configModel(model = self.fine_Tune_model)
        return [feature_extractor, classifier]

    def tuning( self ):
        # weights = trained_featureExtractor.predict(self.data['Support_data'])
        val_data, val_label = self._getValData( )
        optimizer = tf.keras.optimizers.SGD( learning_rate=config.lr, momentum=0.9 )
        self.fine_Tune_model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
        idx = np.random.permutation(len(self.data[ 'Support_data' ]))
        self.fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical(self.data[ 'Support_label' ][ idx ] - np.min(
                                self.data['Support_label' ]),num_classes = self.num_finetune_classes
                        ),
                epochs = 1000,
                shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return self.fine_Tune_model
    def test( self ,isOneShotTask:bool = True):
        nway = 25
        N_test_sample = 1000
        softmax_func = tf.keras.layers.Softmax( )
        correct_count = 0
        test_acc = []
        # load Support and Query dataset
        query_set, query_label = self._getValData( )
        Support_data = self.data[ 'Support_data' ]
        if isOneShotTask:
            '''Perform five shots learning with fine tuning'''
            feature_extractor, classifier = self.loadFineTunedModel( )
            Support_set_embedding = feature_extractor.predict( Support_data )
            for _ in range(N_test_sample):
                Query_data, sample_index = self._getFineTuneTestData( query_set = query_set, nway = nway)
                Query_set_embedding = feature_extractor.predict( Query_data )
                prob_classifier = classifier.predict([Support_set_embedding,Query_set_embedding])
                # sim = cosine_similarity( Support_set_embedding, np.expand_dims(Query_set_embedding[0],axis = 0 ))
                # prob = softmax_func( np.squeeze( sim, -1 ) ).numpy( )
                if np.argmax( prob_classifier ) == sample_index:
                    correct_count += 1
                    print(correct_count)
            acc = (correct_count / N_test_sample) * 100.
            test_acc.append( acc )
            print( "Accuracy %.2f" % acc )
            # Test method 2:
            # optimizer = tf.keras.optimizers.SGD( learning_rate = config.lr, momentum = 0.9 )
            # self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
            # self.fine_Tune_model.evaluate( query_set, query_label )
        if not isOneShotTask:
            '''perform 5 shots learning without fine tuning'''
            five_shot_support_embedding = self._getNShotsEmbedding( Support_data )
            for _ in range(N_test_sample):
                Query_data, sample_index = self._getFineTuneTestData( query_set = query_set, nway = nway )
                Query_set_embedding = self.trained_featureExtractor.predict(Query_data)
                sim = cosine_similarity( five_shot_support_embedding, np.expand_dims( Query_set_embedding[ 0 ], axis = 0 ) )
                prob = softmax_func( np.squeeze( sim, -1 ) ).numpy( )
                if np.argmax( prob ) == sample_index:
                    correct_count += 1
                    print( correct_count )
            acc = (correct_count / N_test_sample) * 100.
            test_acc.append( acc )
            print( "Accuracy %.2f" % acc )
        return test_acc

if __name__ == '__main__':
    # print('start')
    config = getConfig( )
    fineTuningModelObj = fineTuningModel(nshots = 1)
    # fine_Tune_model = fineTuningModelObj.tuning()
    test_acc = fineTuningModelObj.test(isOneShotTask=True)
    print('Done')






