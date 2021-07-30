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
def _getFeatureExtractor():
    _, trained_featureExtractor = defineModel( mode='Alexnet' )
    trained_featureExtractor.load_weights('./models/signFi_wholeModel_weight_AlexNet_training_acc_0.90_on_125cls_user1to4.h5' )
    return trained_featureExtractor
def getEmbeddingVectors(input):
    trained_featureExtractor = _getFeatureExtractor( )
    support_set_embedding = trained_featureExtractor.predict(input)
    return support_set_embedding
def buildFc():
    trained_featureExtractor = _getFeatureExtractor()
    trained_featureExtractor.trainable = False

    input = Input( config.input_shape, name='data input' )
    featureExtractor = trained_featureExtractor( input )
    fc_layer = Dense( units=4096, name='fc_feature' )( featureExtractor )
    fc_layer_out = Dense(units = 5,name = 'fc_out')(fc_layer)
    output = Softmax()(fc_layer_out)

    fc = Model(inputs = input,outputs = output)
    return fc
def fineTuneFC(support_set_embedding,labels):
    fc = buildFc()
    optimizer = tf.keras.optimizers.SGD( learning_rate=config.lr, momentum=0.9 )
    fc.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
    fc.fit(support_set_embedding,labels,epochs = 100,shuffle=True)
    return fc
def getFineTuneData(n_shots:int = 5):
    x_all, y_all = testSign.getFormatedData( source='lab_other' )
    x_all = x_all[1250:1500]
    y_all = y_all[1250:1500]

    start,_ = np.where(y_all == y_all[0])
    fine_tune_data = np.zeros((25,200,60,3))
    fine_tune_labels = np.zeros( (25, 1),dtype = int )
    count = 0
    for i in start[0:n_shots]:
        fine_tune_labels[count:count+n_shots,:] = y_all[i:i+n_shots,:]
        fine_tune_data[count:count+n_shots,:,:,:] = x_all[i:i+n_shots,:,:,:]
        count += n_shots
    return fine_tune_data,fine_tune_labels
def remove_dense(model):
    encoder = Model(inputs=model.input, outputs= model.get_layer('fc_feature').output)
    return encoder
def testFineTune():
    testOneshot = trainTestModel( )
    testSign = signDataLoder( dataDir=config.train_dir )
    encoder = buildFc( )
    encoder = remove_dense(encoder)
    encoder.load_weights('./models/fc_fineTuned.h5')
    encoder.summary()
    x_all, y_all = testSign.getFormatedData( source='lab_other' )
    test_data = x_all[ 1250:1500 ]
    test_labels = y_all[ 1250:1500 ]
    test_acc = testOneshot.signTest( test_data=test_data, test_labels=test_labels,
                                     N_test_sample=1000, embedding_model=encoder,
                                     isOneShotTask=True, mode='fix' )
    return test_acc
class models:
    def __init__( self ):
        pass

    def buildFeatureExtractor( self ,mode:str ='Alexnet'):
        if mode =='Alexnet':
            input = Input( config.input_shape, name = 'input_layer' )
            conv_1 = Conv2D(
                    filters = 96, kernel_size = (11, 5), strides = 2, input_shape = config.input_shape, padding = 'valid',
                    activation = 'relu', name = 'Conv_1'
                    )( input )
            MP_1 = MaxPooling2D( pool_size = 3, strides = 1, name = 'Maxpool_1' )( conv_1 )

            PD_1 = ZeroPadding2D( padding = 2, name = 'Padding_layer_1' )( MP_1 )
            conv_2 = Conv2D( filters = 256, kernel_size = 5, strides = 1, padding = 'valid', name = 'Conv_2' )( PD_1 )
            MP_2 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_2' )( conv_2 )
            Padding_leayer_2 = ZeroPadding2D( padding = 1, name = 'Padding_leayer_2' )( MP_2 )
            Conv_3 = Conv2D(
                    filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                    name = 'Conv_3'
                    )( Padding_leayer_2 )
            Padding_layer_3 = ZeroPadding2D( padding = 1, name = 'Padding_layer_3' )( Conv_3 )
            Conv_4 = Conv2D(
                    filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                    name = 'Conv_4'
                    )( Padding_layer_3 )
            Padding_layer_4 = ZeroPadding2D( padding = 1, name = 'Padding_layer_4' )( Conv_4 )
            Conv_5 = Conv2D(
                    filters = 256, activation = 'relu', kernel_size = (4, 3), strides = 1, padding = 'valid',
                    name = 'Conv_5'
                    )( Padding_layer_4 )
            Maxpool_3 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_3' )( Conv_5 )
            dp = Dropout( 0.5 )( Maxpool_3 )
            ft = Flatten( )( dp )
            FC_1 = Dense( units = 4096, name = 'FC_1' )( ft )
            FC_2 = Dense( units = 4096, name = 'FC_2' )( FC_1 )
            output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ) )( FC_2 )
            feature_extractor = Model( inputs = input, outputs = output )
        return feature_extractor

    def buildTuneModel( self ):
        feature_extractor = self.buildFeatureExtractor( mode = 'Alexnet')
        fc = Dense( units = 25, name = "fine_tune_layer" )( feature_extractor.output )
        output = Softmax( )( fc )
        fine_Tune_model = Model( inputs = feature_extractor.input, outputs = output )
        fine_Tune_model.summary( )
        return fine_Tune_model
class fineTuningModel:
    def __init__( self ):
        self.modelObj = models( )
        self.num_finetune_classes = 25
        self.trained_featureExtractor = self._getFeatureExtractor( )
        self.trained_featureExtractor.trainable = False
        self.classifier = []
        self.input_shape = config.input_shape
        self.trainTestObj = trainTestModel( )
        self.fine_Tune_model = self.buildFineTuningModel( )
        self.lrScheduler = tf.keras.callbacks.LearningRateScheduler( self.trainTestObj.scheduler )
        self.earlyStop = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 20, restore_best_weights =
        True )
        self.data = self.getSQData( )
        self.fineTuned_model_path = config.tunedModel_path
    def getSQData( self ):
        '''For 25 way, 5 shot'''
        testSign = signDataLoder( dataDir=config.train_dir )
        x_all, y_all = testSign.getFormatedData( source='lab_other' )
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
        trained_featureExtractor = self.modelObj.buildFeatureExtractor( mode='Alexnet' )
        trained_featureExtractor.load_weights(config.featureExtractor_path )
        return trained_featureExtractor
    def predict( self, Support_input=None, Query_input = None):
        sim = self.embedding_model.predict( [Support_input,Query_input] )
        return sim
    def buildFineTuningModel( self,weights=None ):
        # input = Input(self.input_shape, name="input_fine_tuning")
        # Support_set_embedding = self.trained_featureExtractor(input)
        # fc = Dense( units = self.num_finetune_classes, name="fine_tune_layer")(Support_set_embedding)
        # output = Softmax( )(fc)
        # fine_Tune_model = Model(inputs = input,outputs = output)
        # fine_Tune_model.summary()
        fine_Tune_model = self.modelObj.buildTuneModel()
        return fine_Tune_model
    def getValData( self ):
        val_data = self.data['Query_data']
        val_label = to_categorical( self.data[ 'Query_label' ] - np.min(self.data['Query_label' ]),num_classes = 25 )
        return [val_data,val_label]
    def tuning( self ):
        # weights = trained_featureExtractor.predict(self.data['Support_data'])
        val_data, val_label = self.getValData()
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
    def _getFineTuneTestData(self,query_set,nway):

        sample_sign = np.random.choice(np.arange(0,125,25),size = 1,replace = False)
        sample_index = random.randint( 0, nway - 1 )
        query_data = np.repeat( query_set[ sample_sign+sample_index ], [ nway ], axis = 0 )
        return [query_data,sample_index]
    def _configModel( self,model ):
        feature_extractor = Model( inputs = model.input, outputs = model.get_layer( 'fine_tune_layer' ).output )

        '''Classifier input: two feature vector
                      output: one probability
        '''
        cls_intput_Support = Input(25,name = 'Support_input')
        cls_intput_Query = Input( 25, name = 'Query_input' )
        cosSim_layer = Dot( axes = 1, normalize = True )([cls_intput_Support,cls_intput_Query])
        cls_output = Softmax( )( tf.squeeze(cosSim_layer,-1) )
        classifier = Model(inputs = [cls_intput_Support,cls_intput_Query],outputs = cls_output)
        return [feature_extractor,classifier]
    def rebuildExtractorFcn(self):
        self.fine_Tune_model.load_weights(config.tunedModel_path)
        feature_extractor, classifier = self._configModel(model = self.fine_Tune_model)
        return [feature_extractor, classifier]
    def getNShotsEmbedding( self ):
        Sign_class = np.arange( 0, 25, 1 )
        Sign_samples = np.arange( 0, 125, 25 )
        five_shot_support_embedding = [ ]
        five_shot_support_data = [ ]
        for i in Sign_class:
            for j in Sign_samples:
                five_shot_support_data.append( Support_data[ i + j ] )
            five_shot_support_embedding.append(
                    np.mean( self.trained_featureExtractor.predict( np.asarray( five_shot_support_data ) ), axis = 0 )
                    )
        five_shot_support_embedding = np.asarray( five_shot_support_embedding )
        return five_shot_support_embedding
    def test( self ,isOneShotTask:bool = True):
        nway = 25
        N_test_sample = 1000
        softmax_func = tf.keras.layers.Softmax( )
        correct_count = 0
        test_acc = []
        # load Support and Query dataset
        query_set, query_label = self.getValData( )
        Support_data = self.data[ 'Support_data' ] # use one sample as support data
        if isOneShotTask:
            feature_extractor, classifier = self.rebuildExtractorFcn( )
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

        return test_acc
                # self.fine_Tune_model.predict()
if __name__ == '__main__':
    # print('start')
    config = getConfig( )
    fineTuningModelObj = fineTuningModel()
    tunedModel = fineTuningModelObj.rebuildExtractorFcn()
    test_acc = fineTuningModelObj.test(isOneShotTask=False)
    print('Done')
    # tunedModel.save_weights(config.tunedModel_path)
    # feature_extractor = buildFeatureExtractor()
    # feature_extractor.load_weights(config.featureExtractor_path)
    # tuneObj = fineTuningModel( )
    # # Tuned_model = tuneObj.tuning()
    # tuneObj.test()
    # Tuned_model.save_weights( tuneObj.fineTuned_model_path )
    # fine_tune_data, fine_tune_labels = getFineTuneData( )
    # Support_input = np.expand_dims(fine_tune_data[1:2],axis=0)
    # Query_input = np.expand_dims(fine_tune_data[6:7],axis=0)
    # sim = fineTuning.predict( Support_input = fine_tune_data[0:2], Query_input = fine_tune_data[5] )





