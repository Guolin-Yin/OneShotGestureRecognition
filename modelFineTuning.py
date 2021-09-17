import tensorflow as tf
import numpy as np
from modelPreTraining import *
from Preprocess.gestureDataLoader import signDataLoader,WidarDataloader
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, Lambda,Dot,concatenate,ZeroPadding2D,Conv2D,\
    MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
from MODEL import models
import matplotlib.pyplot as plt
import random
from scipy.io import savemat,loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from t_SNE import *
class pltConfusionMatrix():
    def __init__( self ):
        pass
    def make_confusion_matrix(self,
            cf,
            group_names = None,
            categories = 'auto',
            count = True,
            percent = True,
            cbar = True,
            xyticks = True,
            xyplotlabels = True,
            sum_stats = True,
            figsize = None,
            cmap = 'Blues',
            title = None
            ):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        '''
        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = [ '' for i in range( cf.size ) ]

        if group_names and len( group_names ) == cf.size:
            group_labels = [ "{}\n".format( value ) for value in group_names ]
        else:
            group_labels = blanks

        if count:
            group_counts = [ "{0:0.0f}\n".format( value ) for value in cf.flatten( ) ]
        else:
            group_counts = blanks

        if percent:
            group_percentages = [ "{0:.2%}".format( value ) for value in cf.flatten( ) / np.sum( cf ) ]
        else:
            group_percentages = blanks

        box_labels = [ f"{v1}{v2}{v3}".strip( ) for v1, v2, v3 in zip( group_labels, group_counts, group_percentages ) ]
        box_labels = np.asarray( box_labels ).reshape( cf.shape[ 0 ], cf.shape[ 1 ] )

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace( cf ) / float( np.sum( cf ) )

            # if it is a binary confusion matrix, show some more stats
            if len( cf ) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[ 1, 1 ] / sum( cf[ :, 1 ] )
                recall = cf[ 1, 1 ] / sum( cf[ 1, : ] )
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                        accuracy, precision, recall, f1_score
                        )
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format( accuracy )
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get( 'figure.figsize' )

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure( figsize = figsize )
        g = sns.heatmap(
                cf, annot = box_labels, fmt = "", cmap = cmap, cbar = cbar, xticklabels = categories,
                yticklabels = categories
                )
        g.set_yticklabels( g.get_yticklabels( ), rotation = 45, fontsize = 12 )
        g.set_xticklabels( g.get_xticklabels( ), rotation = 0, fontsize = 12 )
        if xyplotlabels:
            plt.ylabel( 'True label',fontsize=15  )
            plt.xlabel( 'Predicted label' + stats_text ,fontsize=15 )
        else:
            plt.xlabel( stats_text,fontsize=15 )

        if title:
            plt.title( title,fontsize = 20 )
    def pltCFMatrix( self,y,y_pred,figsize,title ):
        cf_matrix = confusion_matrix(y,y_pred)
        categories = [ 'Push&Pull',
                     'Sweep',
                     'Clap',
                     'Draw-O(Vertical)',
                     'Draw-Zigzag(Vertical)',
                     'Draw-N(Vertical)']
        self.make_confusion_matrix(cf_matrix,categories = categories,figsize = figsize,title=title)
class fineTuningModel:
    def __init__( self,config,nshots = None,isZscore = None, isiheritance = False ):
        self.nshots = config.nshots
        self.isZscore = isZscore
        self.modelObj = models( )
        self.config = config
        self.trainTestObj = PreTrainModel(config = config )
        self.lrScheduler = tf.keras.callbacks.LearningRateScheduler( self.trainTestObj.scheduler )
        self.earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights
        =True, min_delta = 0.0001/2,mode = 'min',verbose=1)
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        self.pretrained_featureExtractor.trainable = True
        if not isiheritance:
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
        testSign = signDataLoader( dataDir = self.config.train_dir )
        fixed_query_set = False
        if type(self.config.source) == list or 'user' in self.config.source:
            _, _, x_test, y_test = testSign.getFormatedData( source = self.config.source,isZscore = self.isZscore )
            x = x_test[ 1250:1500 ]
            y = y_test[ 1250:1500 ]
            num = nshots * 25
            Support_data = x[ 0:num, :, :, : ]
            Support_label = y[0:num,:]
            Query_data = x[num:len(x)+1,:,:,:]
            Query_label = y[num:len(x)+1,:]
        elif 'home' in self.config.source or 'lab' in self.config.source:
            if fixed_query_set:
                _, _, x_test, y_test = testSign.getFormatedData( source = self.config.source, isZscore = self.isZscore )
                num = nshots * self.config.num_finetune_classes
                query_idx = 5 * self.config.num_finetune_classes
                Support_data = x_test[ 0:num, :, :, : ]
                Support_label = y_test[ 0:num, : ]
                Query_data = x_test[ query_idx:len( x_test ) + 1, :, :, : ]
                Query_label = y_test[ query_idx:len( x_test ) + 1, : ]
            else:
                _, _, x_test, y_test = testSign.getFormatedData( source = self.config.source,isZscore=self.isZscore )
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
                self.config.num_finetune_classes
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
            sample_sign = np.random.choice(np.arange(0,len(query_set),self.config.num_finetune_classes),size = 1,replace = False)
            sample_index = random.randint( 0, nway - 1 )
            query_data = np.repeat( query_set[ sample_sign+sample_index ], [ nway ], axis = 0 )
            return [ query_data, sample_index ]
        elif mode == 'random':
            sample_sign = np.random.choice(
                    np.arange( 0, len( query_set ), self.config.num_finetune_classes ), size = 2, replace = False
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
        trained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path )
        return trained_featureExtractor
    def _loadFineTunedModel(self,applyFinetunedModel:bool = True):
        '''
        This function build for load fine tuned model for testing
        :returns pre-trained feature extractor and fine tuned classifier
        '''

        if applyFinetunedModel:
            print( f'loading fine tuned model: {self.config.tunedModel_path}' )
            fine_Tune_model = self.modelObj.buildTuneModel( config = self.config,isTest = True )
            # fine_Tune_model = self.modelObj.buildFeatureExtractor()
            fine_Tune_model.load_weights(self.config.tunedModel_path)
            feature_extractor = fine_Tune_model
            # feature_extractor = Model(
            #         inputs = fine_Tune_model.input, outputs = fine_Tune_model.get_layer( 'fine_tune_layer' ).output
            #         )
        elif not applyFinetunedModel:
            print( f'loading original pretrained feature extractor: {self.config.pretrainedfeatureExtractor_path}' )
            feature_extractor = self.pretrained_featureExtractor
        '''
        Classifier input: two feature vector
                      output: one probability
        '''
        # cls_intput_Support = Input(feature_extractor.output.shape[1],name = 'Support_input')
        # cls_intput_Query = Input( feature_extractor.output.shape[1], name = 'Query_input' )
        cls_intput_Support = Input(4096,name = 'Support_input')
        cls_intput_Query = Input( 4096, name = 'Query_input' )
        cosSim_layer = Dot( axes = 1, normalize = True )([cls_intput_Support,cls_intput_Query])
        cls_output = Softmax( )( tf.squeeze(cosSim_layer,-1) )
        classifier = Model(inputs = [cls_intput_Support,cls_intput_Query],outputs = cls_output)
        # feature_extractor, classifier = self._configModel(model = self.fine_Tune_model)
        return [feature_extractor, classifier]
    def tuning( self ,init_weights = True,init_bias = False):
        # self.pretrained_featureExtractor = _getPreTrainedFeatureExtractor()
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
                bias = np.tile(np.mean(-np.sum( p * np.log(p ),axis = 1 ) ),self.config.num_finetune_classes)
            else:
                bias = np.zeros(self.config.num_finetune_classes)
            fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )

        val_data, val_label = self._getValData( )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
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
                                self.data[ 'Support_label' ]),num_classes = self.config.num_finetune_classes),
                epochs = 1000,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return fine_Tune_model
    def test( self, applyFinetunedModel:bool=True ):
        # self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        # self.pretrained_featureExtractor.trainable = True
        nway = self.config.num_finetune_classes
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
class fineTuningWidar(fineTuningModel):
    def __init__( self,config,isMultiDomain:bool = False ):
        super().__init__(config = config, isiheritance = True, )
        self.WidarDataLoaderObj = WidarDataloader(config = config)
        # self.selected_gesture_samples_data,self.x,self.y = self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.y
        self.config = config
        self.nshots = config.nshots
        self.nshots_per_domain = config.nshots_per_domain
        # self.nshots_per_domain = int(self.nshots/5)
        self.nways = config.num_finetune_classes
        self.initializer = tf.keras.initializers.RandomUniform( minval = 0., maxval = 1. )
        self.fine_Tune_model = self.modelObj.buildTuneModel(
                pretrained_feature_extractor = self.pretrained_featureExtractor,
                isTest = False, config = self.config
                )
        if isMultiDomain:
            self.WidarDataLoaderObjMulti = WidarDataloader(
                    isMultiDomain = isMultiDomain,
                    config = config
                    )
        else:
            self.WidarDataLoaderObj = WidarDataloader(
                     isMultiDomain = isMultiDomain,
                    config = config
                    )
    # def getMultiDomainData(self,isTest=False):
    #     Support_data = []
    #     Support_label =[]
    #     Query_data=[]
    #     Query_label=[]
    #     Val_data=[]
    #     Val_label=[]
    #     record=[]
    #     for i in range(1,6):
    #         self.WidarDataLoaderObjMulti = WidarDataloader(
    #                 dataDir = config.train_dir, selection = (2, i, 3),
    #                 config = config
    #                 )
    #         if not isTest:
    #             data = self.WidarDataLoaderObjMulti.getSQDataForTest(
    #                     nshots = 1, mode = 'fix',
    #                     isTest = isTest,
    #                     # Best = config.record[ i - 1 ]
    #                     )
    #         else:
    #             data = self.WidarDataLoaderObjMulti.getSQDataForTest(
    #                     nshots = 1, mode = 'fix',
    #                     isTest = isTest,
    #                     Best = (config.record.reshape(5,6,1))[i-1]
    #                     )
    #         Support_data.append(data['Support_data'])
    #         Support_label.append( data[ 'Support_label' ] )
    #         Query_data.append( data[ 'Query_data' ] )
    #         Query_label.append( data[ 'Query_label' ] )
    #         Val_data.append( data[ 'Val_data' ] )
    #         Val_label.append( data[ 'Val_label' ] )
    #         record.append( data[ 'record' ])
    #     Support_data_out = np.zeros((30,200,60,3))
    #     Support_label_out = np.zeros((30,1),dtype= int)
    #     idx = 0
    #     for d in range(config.num_finetune_classes):
    #         for j in range(len(Support_data)):
    #             Support_data_out[ idx, :, :, : ] = Support_data[ j ][ d, :, :, : ]
    #             Support_label_out[idx] = Support_label[ j ][ d ]
    #             idx += 1
    #     output = {
    #             'Support_data' : Support_data_out,
    #             'Support_label': Support_label_out,
    #             'Query_data'   : np.concatenate(Query_data,axis = 0),
    #             'Query_label'  : np.concatenate(Query_label,axis = 0),
    #             'Val_data'     : np.concatenate(Val_data,axis = 0),
    #             'Val_label'    : np.concatenate(Val_label,axis = 0),
    #             'record'       : np.concatenate(record,axis = 0)
    #             }
    #     return output
    def _getNShotsEmbedding( self,feature_extractor,Support_set ):
        Support_set_embedding_all = feature_extractor.predict( Support_set )
        Support_set_embedding = []
        for i in range(self.nways):
            Support_set_embedding.append(np.mean(Support_set_embedding_all[i*self.nshots:i*self.nshots+self.nshots],
                    axis=0))
        return np.asarray(Support_set_embedding)
    def tuning( self,init_weights = True,init_bias = False, isLoopSearch:bool = False, iteration: int = None,isTest:bool = False):
        self.data = self.WidarDataLoaderObj.getSQDataForTest(
                nshots = self.nshots, mode = 'fix',
                isTest = isTest,
                Best = None
                )
        # self.data = self.WidarDataLoaderObjMulti.getMultiDomainSQDataForTest(
        #         nshots_per_domain = self.nshots_per_domain, isTest = False
        #         )
        self.pretrained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        if init_weights:
            if self.nshots == 1:
                weights = np.transpose( self.pretrained_featureExtractor.predict( self.data[ 'Support_data' ] ) )
            elif self.nshots > 1:
                weights = np.transpose(
                        self._getNShotsEmbedding( self.pretrained_featureExtractor, self.data[ 'Support_data' ] )
                        )
            if init_bias:
                p = self.fine_Tune_model.predict( self.data[ 'Query_data' ] )
                bias = np.tile( np.mean( -np.sum( p * np.log( p ), axis = 1 ) ), self.config.num_finetune_classes )
            else:
                bias = np.zeros( self.config.num_finetune_classes )
            self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
            # self.config.weight = self.fine_Tune_model.get_layer( 'FC_1' ).get_weights( )
        val_data, val_label = self.data[ 'Val_data' ], to_categorical(
                self.data[ 'Val_label' ], num_classes
                = self.config.num_finetune_classes
                )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )
        # optimizer = tf.keras.optimizers.SGD(
        #         learning_rate = config.lr,
        #         momentum = 0.99,
        #         )
        # optimizer = tf.keras.optimizers.Adadelta(
        #         learning_rate = config.lr, rho = 0.50, epsilon = 1e-06, name = 'Adadelta',
        #
        #         )
        # optimizer =  tf.keras.optimizers.RMSprop(
        #                                     learning_rate=config.lr,
        #                                     rho=0.99, momentum=0.9,
        #                                     epsilon=1e-06,
        #                                     centered=False,
        #                                     name='RMSprop',
        #                                                     )
        # optimizer = tf.keras.optimizers.Adamax(
        #                                          learning_rate=self.config.lr, beta_1=0.90, beta_2=0.98, epsilon=1e-08,
        #                                          name='Adamax'
        #                                      )
        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        idx = np.random.permutation( len( self.data[ 'Support_data' ] ) )
        history = self.fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical(self.data[ 'Support_label' ][ idx ] , num_classes
                = self.config.num_finetune_classes),
                epochs = 1000,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,self.data['record'],history]
    def _getoutput( self, feature_extractor ):
        return Model(inputs = feature_extractor.input,outputs = feature_extractor.get_layer('lambda_layer').output)
    def test( self, applyFinetunedModel:bool,isFineTunedModel:bool):
        self.feature_extractor, self.classifier = self._loadFineTunedModel(
                applyFinetunedModel = applyFinetunedModel
                )
        print(f'check for {self.nshots} shots '
              f'accuracy......................................................................')
        N_test_sample = 1200
        correct_count = 0
        test_acc = [ ]
        y_true = []
        y_pred = []
        label_true = []
        n = 6
        feature_extractor = self._getoutput(self.feature_extractor)
        classifier = self.classifier
        # Support_set_embedding = np.transpose(self.feature_extractor.get_layer('fine_tune_layer').get_weights()[0])
        # Support_set_embedding = feature_extractor.predict()
        for i in range( N_test_sample ):
            self.data = self.WidarDataLoaderObj.getSQDataForTest(
                    nshots = self.nshots, mode = 'fix',
                    isTest = True, Best = self.config.record
                    )
            # self.data = self.WidarDataLoaderObjMulti.getMultiDomainSQDataForTest(
            #         nshots_per_domain = self.nshots_per_domain, isTest = True, Best = config.record
            #         )
            # Support_set_embedding = matrix
            Query_set = self.data['Query_data']
            Support_set = self.data['Support_data']
            if isFineTunedModel:
                Support_set_embedding = np.transpose(
                        self.feature_extractor.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                        )
            else:
                Support_set_embedding = self._getNShotsEmbedding(feature_extractor,Support_set)
            gesture_type_idx = i%6
            Query_sample = np.repeat( np.expand_dims( Query_set[ gesture_type_idx ], axis = 0 ), n, axis = 0 )
            Query_set_embedding = feature_extractor.predict( Query_sample )
            # model = self._getoutput( feature_extractor )
            prob_classifier = classifier.predict( [ Support_set_embedding, Query_set_embedding ] )
            y_true.append(gesture_type_idx)
            label_true.append(self.data['Query_label'][gesture_type_idx][0])
            y_pred.append( np.argmax(prob_classifier))
            if np.argmax( prob_classifier ) == gesture_type_idx:
                correct_count += 1
            print( f'The number of correct: {correct_count}, The number of test count {i}' )
        acc = (correct_count / N_test_sample) * 100.
        test_acc.append( acc )
        print( "Accuracy %.2f" % acc )
        return test_acc,[y_true,y_pred],label_true
def searchBestSample(config):
    config.nshots_per_domain = None
    # config.nshots = int(5*1*1*config.nshots_per_domain)
    config.nshots = 5
    config.train_dir = 'E:/Sensing_project/Cross_dataset/20181109/User1'
    # config.train_dir = 'E:/Cross_dataset/20181115'
    config.num_finetune_classes = 6
    config.lr = 1e-4
    config.domain_selection = (2, 2, 3)
    config.pretrainedfeatureExtractor_path = \
        './models/signFi_featureExtractor_weight_AlexNet_lab_training_acc_0.95_on_250cls.h5'
    fineTuningWidarObj = fineTuningWidar(config = config,isMultiDomain = False)
    location,orientation,Rx = config.domain_selection
    val_acc = 0
    acc_record = []
    for i in range(100):
        fine_Tune_model,record,history = fineTuningWidarObj.tuning(isLoopSearch = True, init_bias = True,
                iteration = i)
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        acc_record.append(history.history[ 'val_acc' ][ -1 ])
        if val_acc < history.history['val_acc'][-1]:
            val_acc = history.history[ 'val_acc' ][ -1 ]
            config.tunedModel_path = f'./models/Publication_related/widar_fineTuned_model_20181109' \
                                     f'_{config.nshots}shots_' \
                                     f'_domain{config.domain_selection}_{val_acc:0.2f}.h5'
            fine_Tune_model.save_weights(config.tunedModel_path)
            best_record = record
            config.record = best_record
            print(f'Updated record is: {best_record}')
            mdic = {'record':best_record,
                    'val_acc':val_acc}
            config.setMatSavePath(f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots"
                                  f"_domain_{config.domain_selection}_20181109.mat")
            savemat( config.matPath, mdic )
            if val_acc >= 0.900:
                print(f'reached expected val_acc {val_acc}')
                break
    config.getSampleIdx( )
    test_acc,[y_true,y_pred],label_true = fineTuningWidarObj.test(applyFinetunedModel = True,isFineTunedModel = True)
    plt_cf = pltConfusionMatrix( )
    title = f'{config.nshots}_shot_sRx_{Rx}_domain_{config.domain_selection}'
    plt_cf.pltCFMatrix( y = label_true, y_pred = y_pred, figsize = (18,15),title = title )
    print(f'The average accuracy is {np.mean(acc_record)}')
    plt.savefig(f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/results_figs/{config.nshots}shots_'
                f'{config.domain_selection}_finetuned.png')
    return acc_record
def evaluation( domain_selection,isMultiDomain ):
    # config.getFineTunedModelPath( )
    # location, orientation, Rx = config.domain_selection
    config = getConfig( )
    config.domain_selection = domain_selection
    config.nshots = 2
    config.pretrainedfeatureExtractor_path = \
            './models/signFi_featureExtractor_weight_AlexNet_lab_training_acc_0.95_on_250cls.h5'
    # config.setMatSavePath(
    #         f"./Sample_index/Publication_related/sample_index_record_for_2_shots_domain_(2, 2, 3)_20181109.mat"
    #         )
    config.matPath = f"./Sample_index/Publication_related/sample_index_record_for_2_shots_domain_(2, 2, 3)_20181109.mat"
    config.getSampleIdx( )
    config.train_dir = 'E:/Cross_dataset/20181109/User1'
    # config.tunedModel_path = f'./models/fine_tuning_widar/widar_fineTuned_model_20181109_1shots_test_domain_(2, 2, 3).h5'
    config.tunedModel_path = './models/Publication_related/widar_fineTuned_model_20181109_2shots__domain(2, 2, 3)_.h5'
    config.record = loadmat(config.matPath)['record']
    config.domain_selection = domain_selection
    config.num_finetune_classes = 6
    fineTuneModelEvalObj = fineTuningWidar( config = config, isMultiDomain = isMultiDomain )
    test_acc,[y_true,y_pred],label_true = fineTuneModelEvalObj.test(applyFinetunedModel =True,isFineTunedModel = False)
    plt_cf = pltConfusionMatrix( )
    plt_cf.pltCFMatrix(
            y = label_true, y_pred = y_pred, figsize = (18, 15), title = f'{config.nshots}_shots '
                                                                         f'domain_{domain_selection}'
                                                                         f'with_fine_'
                                                                         f'tuning_results'
            )
def compareDomain():
    config = getConfig( )
    config.nshots_per_domain = 2
    config.nshots = int( 5 * 1 * 1 * config.nshots_per_domain )
    config.train_dir = 'E:/Cross_dataset/20181109/User1'
    config.num_finetune_classes = 6
    config.lr = 1e-3
    config.domain_selection = (2, 2, 3)
    config.pretrainedfeatureExtractor_path = \
        './models/feature_extractor_weight_Alexnet_lab_250cls_val_acc_0.996_no_zscore.h5'
    config.tunedModel_path = \
        f'./models/MultiDomain_Widar' \
        f'/widar_fineTuned_model_20181109_10shots_MultiDomainOrientation_Rx3_location_2_multi_csiANP_.h5'
    fineTuningWidarObj = fineTuningWidar( config = config, isMultiDomain = True )
    feature_extractor, classifier = fineTuningWidarObj._loadFineTunedModel(
            applyFinetunedModel = True
            )
    feature_extractor = fineTuningWidarObj._getoutput( feature_extractor )

    WidarDataLoaderObj223 = WidarDataloader(
            dataDir = config.train_dir, selection = (2,2,3), isMultiDomain = False,
            config = config
            )
    data223 = WidarDataLoaderObj223.getSQDataForTest(
            nshots = 1, mode = 'fix',
            isTest = False
            )
    WidarDataLoaderObj233 = WidarDataloader(
            dataDir = config.train_dir, selection = (3, 5, 1), isMultiDomain = False,
            config = config
            )
    data233 = WidarDataLoaderObj233.getSQDataForTest(
            nshots = 1, mode = 'fix',
            isTest = False
            )
    pred_223 = feature_extractor.predict(data223['Val_data'])
    label_223 = data223['Val_label']
    pred_233 = feature_extractor.predict( data233['Val_data'] )
    label_233 = data233[ 'Val_label' ]
    class_t_sne( pred_223, label_223,perplexity = 40, n_iter = 3000 )
if __name__ == '__main__':
    config = getConfig( )

    acc_record = searchBestSample(config)

    '''Testing with specific domain selection'''
    # evaluation(
    #         domain_selection = (2, 2, 3),
    #         isMultiDomain = False
    #         )
    # compareDomain()





