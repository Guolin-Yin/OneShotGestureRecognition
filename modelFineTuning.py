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
# from MODEL import models
import matplotlib.pyplot as plt
import random
from scipy.io import savemat,loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from methodTesting.t_SNE import *
# from modelPreTraining import PreTrainModel
class pltConfusionMatrix():
    def __init__( self ):
        pass
    def make_confusion_matrix(self,
            cf,
            group_names = None,
            categories = 'auto',
            count = True,
            percent = False,
            cbar = True,
            xyticks = True,
            xyplotlabels = True,
            sum_stats = True,
            figsize = None,
            cmap = 'Oranges',
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
                # stats_text = "\n\nAccuracy={:0.3f}".format( accuracy )
                stats_text = ""
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
                yticklabels = categories,annot_kws={"size": 18},
                )
        g.set_yticklabels( g.get_yticklabels( ), rotation = 45, fontsize = 22 )
        g.set_xticklabels( g.get_xticklabels( ), rotation = 0, fontsize = 22 )
        if xyplotlabels:
            plt.ylabel( 'True label',fontsize=22  )
            plt.xlabel( 'Predicted label' + stats_text ,fontsize=22 )
        else:
            plt.xlabel( stats_text,fontsize=22 )

        # if title:
        #     plt.title( title,fontsize = 20 )
    def pltCFMatrix( self,y,y_pred,figsize,title ):
        cf_matrix = confusion_matrix(y,y_pred)
        categories = [ 'P&P',
                     'Sweep',
                     'Clap',
                     'O',
                     'Zigzag',
                     'N']
        self.make_confusion_matrix(cf_matrix,categories = categories,figsize = figsize,title=title)
class fineTuningSignFi:
    def __init__( self,config, isZscore = False, isiheritance = False ):
        self.nshots = config.nshots
        self.isZscore = isZscore
        self.modelObj = models( )
        self.config = config
        self.trainTestObj = PreTrainModel(config = config )
        self.lrScheduler = tf.keras.callbacks.LearningRateScheduler( self.trainTestObj.scheduler )
        self.earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights
        =True, min_delta = 0.0001/2,mode = 'min',verbose=1)
        # self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        # self.pretrained_featureExtractor.trainable = True
        if self.nshots == 1:
            self.isOneShotTask = True
        else:
            self.isOneShotTask = False
    def _getSQData( self,nshots:int ):
        '''
        This function build for split support set and query set according to the number of shots
        :param nshots:
        :return:
        '''
        testSign = signDataLoader( config = self.config )
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
                num = nshots * self.config.N_novel_classes
                query_idx = 5 * self.config.N_novel_classes
                Support_data = x_test[ 0:num, :, :, : ]
                Support_label = y_test[ 0:num, : ]
                Query_data = x_test[ query_idx:len( x_test ) + 1, :, :, : ]
                Query_label = y_test[ query_idx:len( x_test ) + 1, : ]
            else:
                _, _, x_test, y_test = testSign.getFormatedData( source = self.config.source,isZscore=self.isZscore )
                num = nshots * (np.max(y_test) + 1 - np.min(y_test))
                Support_data = x_test[ 0:num, :, :, : ]
                Support_label = y_test[ 0:num, : ]
                Query_data = x_test[ num:len( x_test ) + 1, :, :, : ]
                Query_label = y_test[ num:len( x_test ) + 1, : ]
        output = {'Support_data':Support_data,
                  'Support_label':Support_label,
                  'Query_data':Query_data,
                  'Query_label':Query_label}
        return output
    def _getValData( self,Query_data,Query_label ):
        '''
        Get the validation data for fine tuning
        :return:
        '''
        # val_data = self.data['Query_data']
        # val_label = to_categorical(
        #         self.data[ 'Query_label' ] - np.min( self.data[ 'Query_label' ] ), num_classes =
        #         self.config.N_novel_classes
        #         )
        val_data = Query_data
        val_label = to_categorical(
                Query_label - np.min( Query_label ), num_classes =
                self.config.N_novel_classes
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
            sample_sign = np.random.choice(np.arange(0,len(query_set),self.config.N_novel_classes ),size = 1,
                    replace = False)
            sample_index = random.randint( 0, nway - 1 )
            query_data = np.repeat( query_set[ sample_sign+sample_index ], [ nway ], axis = 0 )
            return [ query_data, sample_index ]
        elif mode == 'random':
            sample_sign = np.random.choice(
                    np.arange( 0, len( query_set ), self.config.N_novel_classes ), size = 2, replace = False
                    )
            sample_index = random.randint( 0, nway - 1 )
            support_data = np.repeat( query_set[ sample_sign[0] + sample_index ], [ nway ], axis = 0 )
            query_data = np.repeat( query_set[ sample_sign[1] + sample_index ], [ nway ], axis = 0 )
            return [ support_data,query_data, sample_index ]
    def _getNShotsEmbedding( self,featureExtractor, Support_data):
        Sign_class = np.arange( 0, self.config.N_novel_classes, 1 )
        # Sign_samples = np.arange( 0, 125, 25 )
        Sign_samples = np.arange( 0, len(Support_data), self.config.N_novel_classes )
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
    def _loadFineTunedModel(self,applyFinetunedModel:bool = True, useWeightMatrix:bool = False,isSepRx:bool = False):
        '''
        This function build for load fine tuned model for testing
        :returns pre-trained feature extractor and fine tuned classifier
        '''
        './models/fine_tuning_signfi/fc_fineTuned_250Cls_labTohome_1_shot_without_Zscore.h5'
        if applyFinetunedModel:

            fine_Tune_model = self.modelObj.buildTuneModel( config = self.config,isTest = True )
            if not isSepRx:
                print( f'loading fine tuned model: {self.config.tunedModel_path}' )
                print('loading fine tuned model weights}')
                fine_Tune_model.load_weights(self.config.tunedModel_path)
            if useWeightMatrix:
                feature_extractor = fine_Tune_model
            else:
                feature_extractor = Model(
                        inputs = fine_Tune_model.input, outputs = fine_Tune_model.get_layer( 'lambda_layer' ).output
                        )
        elif not applyFinetunedModel:
            print( f'loading original pretrained feature extractor: {self.config.pretrainedfeatureExtractor_path}' )
            feature_extractor = self.modelObj.buildFeatureExtractor(mode = 'Alexnet')
            feature_extractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        '''
        Classifier input: two feature vector
                      output: one probability
        '''
        # cls_intput_Support = Input(feature_extractor.output.shape[1],name = 'Support_input')
        # cls_intput_Query = Input( feature_extractor.output.shape[1], name = 'Query_input' )
        cls_intput_Support = Input(1280,name = 'Support_input')
        cls_intput_Query = Input( 1280, name = 'Query_input' )
        cosSim_layer = Dot( axes = 1, normalize = True )([cls_intput_Support,cls_intput_Query])
        cls_output = Softmax( )( tf.squeeze(cosSim_layer,-1) )
        classifier = Model(inputs = [cls_intput_Support,cls_intput_Query],outputs = cls_output)
        # feature_extractor, classifier = self._configModel(model = self.fine_Tune_model)
        return [feature_extractor, classifier]
    def tuning( self ,init_weights = True,init_bias = False):
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        self.pretrained_featureExtractor.trainable = True
        self.data = self._getSQData( nshots = self.nshots )
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
                bias = np.tile(np.mean(-np.sum( p * np.log(p ),axis = 1 ) ),self.config.N_novel_classes )
            else:
                bias = np.zeros(self.config.N_novel_classes )
            fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )

        val_data, val_label = self._getValData(self.data['Query_data'],self.data['Query_label'] )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                epsilon = 1e-06,
                )
        # optimizer = tf.keras.optimizers.SGD(
        #         learning_rate = self.config.lr,
        #         momentum = 0.9,
        #         )
        fine_Tune_model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
        idx = np.random.permutation(len(self.data[ 'Support_data' ]))
        fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical( self.data[ 'Support_label' ][ idx ] - np.min(
                                self.data[ 'Support_label' ]),num_classes = self.config.N_novel_classes ),
                epochs = 1000,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return fine_Tune_model
    def _mapToNways(self,Support_data,query_set,query_label,nway):
        query_label = np.argmax( query_label, axis = 1 )
        label = np.unique(query_label)
        selected_sign = np.unique(np.random.choice(label,size = nway,replace=False))
        Support_data = Support_data[selected_sign,:,:,:]
        # selected_data_idx = np.where(query_label == selected_sign)
        index = np.random.choice(np.arange(0,len(query_set),self.config.N_novel_classes ),size = 1,
                    replace = False)
        query_data = query_set[index+selected_sign,:,:,:]
        sample_index = random.randint( 0, nway - 1 )
        Query_data = np.repeat(np.expand_dims(query_data[sample_index],axis=0),[nway],axis = 0)
        return [ Support_data, Query_data,sample_index]
    def test( self, nway,applyFinetunedModel:bool=True ):
        # self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        # self.pretrained_featureExtractor.trainable = False
        # softmax_func = tf.keras.layers.Softmax( )
        self.data = self._getSQData( nshots = self.nshots )
        N_test_sample = 100
        feature_extractor, classifier = self._loadFineTunedModel( applyFinetunedModel )
        # load Support and Query dataset
        query_set, query_label = self._getValData(self.data['Query_data'],self.data['Query_label'] )
        Support_data = self.data[ 'Support_data' ]
        Support_label = self.data[ 'Support_label' ]
        test_acc = [ ]
        for i in range( 24,25 ):
            nway = i
            correct_count = 0
            print( f'................................Checking {nway} ways accuracy................................' )
            if self.isOneShotTask:
                for i in range(N_test_sample):
                    Selected_Support_data, Selected_Query_data, sample_index = self._mapToNways(Support_data,query_set,
                            query_label,nway)
                    Support_set_embedding = feature_extractor.predict( Selected_Support_data )
                    # Query_data, sample_index = self._getDataToTesting( query_set = query_set, nway = nway )
                    Query_set_embedding = feature_extractor.predict( Selected_Query_data )
                    prob_classifier = classifier.predict([Support_set_embedding,Query_set_embedding])
                    if np.argmax( prob_classifier ) == sample_index:
                        correct_count += 1
                        print( f'The number of correct: {correct_count}, The number of test count {i}' )
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
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
class fineTuningWidar(fineTuningSignFi ):
    def __init__( self,config,isMultiDomain:bool = False,isiheritance=False ):
        super().__init__(config = config, isiheritance = True, )
        self.isMultiDomain = isMultiDomain
        if not isiheritance:
            self.WidarDataLoaderObj = WidarDataloader(config = config, isMultiDomain = True )
        # self.selected_gesture_samples_data,self.x,self.y = self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.y
            self.config = config
            self.nshots = config.nshots
            self.nshots_per_domain = config.nshots_per_domain
            # self.nshots_per_domain = int(self.nshots/5)
            self.nways = config.N_novel_classes
            self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
            self.pretrained_featureExtractor.trainable = True
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
    #     for d in range(config.N_novel_classes):
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
    def _getoutput( self, feature_extractor ):
        return Model(inputs = feature_extractor.input,outputs = feature_extractor.get_layer('lambda_layer').output)
    def _getNShotsEmbedding( self,feature_extractor,Support_set ):
        Support_set_embedding_all = feature_extractor.predict( Support_set )
        Support_set_embedding = []
        if self.isMultiDomain:
            n = len(self.config.domain_selection)
            for i in range( self.nways ):
                Support_set_embedding.append(
                        np.mean(
                                Support_set_embedding_all[ i * n * self.nshots:i * n * self.nshots + n * self.nshots ],
                                axis = 0
                                )
                        )
        else:
            # n=self.nshots
            for i in range( self.nways ):
                Support_set_embedding.append(
                        np.mean(
                                Support_set_embedding_all[ i * self.nshots:i * self.nshots + self.nshots ],
                                axis = 0
                                )
                        )

        return np.asarray(Support_set_embedding)
    def tuning( self,init_weights = True,init_bias = False,isTest:bool = False):
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
                bias = np.tile( np.mean( -np.sum( p * np.log( p ), axis = 1 ) ), self.config.N_novel_classes )
            else:
                bias = np.zeros( self.config.N_novel_classes )
            self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
            # self.config.weight = self.fine_Tune_model.get_layer( 'FC_1' ).get_weights( )
        val_data, val_label = self.data[ 'Val_data' ], to_categorical(
                self.data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
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
                = self.config.N_novel_classes ),
                epochs = 1000,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,self.data['record'],history]
    def tuningMultiRx( self ):
        data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( nshots_per_domain = self.config.nshots, isTest = False )
        Support_data = [ ]
        keys = list( data[ 'Support_data' ].keys( ) )
        [ Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 ) ) for j in
          range( len( keys ) ) ]
        s_data_array = np.concatenate( Support_data, axis = 0 )
        self.pretrained_featureExtractor.load_weights( self.config.pretrainedfeatureExtractor_path )
        weights = np.transpose(
                self._getNShotsEmbedding( self.pretrained_featureExtractor, s_data_array )
                )
        bias = np.zeros( self.config.N_novel_classes )
        self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )
        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        val_data, val_label = data[ 'Val_data' ], to_categorical(
                data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )

        idx = np.random.permutation( len( s_data_array ) )
        history = self.fine_Tune_model.fit(
                s_data_array[ idx ],
                to_categorical(data[ 'Support_label' ][ idx ], num_classes = self.config.N_novel_classes),
                epochs = 1000,
                verbose = 0,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,data['record'],history]
    def test( self, applyFinetunedModel:bool):
        self.feature_extractor, self.classifier = self._loadFineTunedModel(
                applyFinetunedModel = applyFinetunedModel,useWeightMatrix = True
                )
        print(f'check for {self.nshots} shots '
              f'accuracy......................................................................')
        N_test_sample = 600
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
            if applyFinetunedModel:
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
    def testMultiRx( self,applyFinetunedModel:bool,useWeightMatrix:bool = True,n_Rx:int = 6):
        # n_Rx = 5
        self.feature_extractor, _ = self._loadFineTunedModel(
                applyFinetunedModel = applyFinetunedModel,useWeightMatrix = useWeightMatrix)
        print(f'check for {self.nshots} shots {n_Rx} Receivers'
              f'accuracy......................................................................')
        N_test_sample,correct_count,test_acc,y_true,y_pred,label_true, n = 600,0,[],[],[],[],6
        softmax_func = tf.keras.layers.Softmax( )
        if useWeightMatrix:
            weights = np.transpose(
                    self.feature_extractor.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                    )
            feature_extractor = self._getoutput( self.feature_extractor )
        for i in range( N_test_sample ):
            Query_data_selection = []
            data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( 1, False )
            Support_data,Query_data,Support_embedding,Query_embedding,sim,sim_mean = [],[],[],[],[],[]
            keys = list( data[ 'Support_data' ].keys( ) )
            [Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 )) for j in range(len(keys))]
            # random selection of specific receiver
            selected_Rx_idx = np.unique(np.random.choice(np.arange(0,6),n_Rx,replace = False))
            # selection of specific receiver
            # selected_Rx_idx = np.arange(0,n_Rx)
            for g in range(len(Support_data)):
                Support_data[g] = Support_data[g][selected_Rx_idx]
            if not useWeightMatrix:
                feature_extractor = self.feature_extractor
                '''gestures, 6 receivers CSI embedding'''
                [Support_embedding.append(feature_extractor.predict(Support_data[j])) for j in range(len(Support_data))]
            g_idx = np.random.choice( np.arange( 0, len( keys ) ), 1, replace = False )[ 0 ]
            sample_idx = np.random.choice( np.arange( 0, len(data['Query_data'][keys[g_idx]][0]) ), 1,replace = False)[ 0 ]
            buffer = data['Query_data'][keys[g_idx]]
            '''select one sample for the selected gesture, corresponding to six receivers'''
            [Query_data.append(buffer[ant][sample_idx]) for ant in range(len(data['Query_data'][keys[g_idx]]))]
            [Query_data_selection.append(Query_data[ant]) for ant in selected_Rx_idx]
            Query_embedding.append( feature_extractor.predict( np.asarray( Query_data_selection ) ) )
            if useWeightMatrix:
                p = []
                buf_sim = [ cosine_similarity( weights, np.expand_dims( Query_embedding[ 0 ][ ant ], axis = 0 ) )
                            for ant in range( len(Query_embedding[0])) ]
                [p.append(np.expand_dims(softmax_func( np.squeeze(buf_sim[tt],axis=-1) ).numpy( ),axis = 0 )) for tt in range(len(buf_sim) )]
                p = np.sum(np.concatenate( p, axis = 0 ),axis=0)
            else:
                for g_sim in range(len(Support_embedding ) ):
                    [sim.append(cosine_similarity( np.expand_dims(Support_embedding[ g_sim ][ ant ],axis=0 ),np.expand_dims(
                            Query_embedding[0][ant],axis=0) )) for ant in range(len(Query_embedding[0]))]
                    sim_mean.append(np.mean(sim))
                p = softmax_func(sim_mean).numpy( )
            if np.argmax( p ) == g_idx:
                correct_count += 1
        acc = (correct_count / N_test_sample) * 100.
        print( "Accuracy %.2f" % acc )
        return acc
    def GetparamForMulRx(self,path_idx:int):
        FE, _ = self._loadFineTunedModel(
                applyFinetunedModel = True, useWeightMatrix = True, isSepRx = True
                )
        FE.load_weights( self.config.tunedModel_path[ path_idx ] )
        feature_extractor = self._getoutput( FE )
        weights = np.transpose(
                FE.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                )
        return feature_extractor,weights
    def testMultiRxSep( self, N_Rx:int ):
        f1, w1 = self.GetparamForMulRx( path_idx = 0 )
        f2, w2 = self.GetparamForMulRx( path_idx = 1 )
        f3, w3 = self.GetparamForMulRx( path_idx = 2 )
        f4, w4 = self.GetparamForMulRx( path_idx = 3 )
        f5, w5 = self.GetparamForMulRx( path_idx = 4 )
        f6, w6 = self.GetparamForMulRx( path_idx = 5 )
        feature_extractor = [f1,f2,f3,f4,f5,f6]
        weights=[w1,w2,w3,w4,w5,w6]
        N_test_sample, correct_count, test_acc, y_true, y_pred, label_true, n = 600, 0, [ ], [ ], [ ], [ ], 6
        print(self.config.tunedModel_path)
        softmax_func = tf.keras.layers.Softmax( )
        print(f'check for {self.nshots} shots, {N_Rx}_Receivers '
              f'accuracy......................................................................')
        for i in range( N_test_sample ):
            data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( self.config.nshots, True,
                    Best = self.config.record )
            Support_data, Query_data, Support_embedding, Query_embedding, sim, sim_mean,p = [ ], [ ], [ ], [ ], [ ], \
                                                                                            [ ],[ ]
            keys = list( data[ 'Support_data' ].keys( ) )
            '''gestures, 6 receivers CSI data'''
            # [ Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 ) ) for j in range(len( keys ) ) ]
            # '''gestures, 6 receivers CSI embedding'''
            # for j in range( len( Support_data ) ):
            #     data_buf = Support_data[ j ]
            #     emb_buf = []
            #     for f_idx in range( len( data_buf)):
            #         emb_buf.append(feature_extractor[f_idx].predict( np.expand_dims(data_buf[f_idx],axis=0) ))
            # Support_embedding.append( np.concatenate(emb_buf,axis=0) )
            g_idx = np.random.choice( np.arange( 0, len( keys ) ), 1, replace = False )[ 0 ]
            sample_idx = np.random.choice( np.arange( 0, len(data['Query_data'][keys[g_idx]][0]) ), 1,replace = False)[ 0 ]
            buffer = data['Query_data'][keys[g_idx]]
            ant_selection = np.random.choice( np.arange( 0, 6), N_Rx, replace = False)
            '''select one sample for the selected gesture, corresponding to six receivers'''
            [Query_data.append(buffer[ant][sample_idx]) for ant in range(len(data['Query_data'][keys[g_idx]]))]
            '''select antenna'''
            Query_data_selection = []
            weights_selection = []
            for at in ant_selection:
                Query_data_selection.append(Query_data[at])
                weights_selection.append(weights[at])

            [Query_embedding.append(feature_extractor[ant_selection[ant]].predict(np.expand_dims(
                    Query_data_selection[ant],axis = 0))) for ant in range(len(Query_data_selection))]
            for w in range(len(Query_embedding)):
                sim.append(cosine_similarity( weights_selection[w], Query_embedding[ w ]) )
            for g in range( len( sim ) ):
                p.append(softmax_func(  np.squeeze(sim[ g ] )).numpy())
            p = np.sum( np.asarray(p), axis = 0 )
            if np.argmax( p ) == g_idx:
                correct_count += 1
        acc = (correct_count / N_test_sample) * 100.
        print( "Accuracy %.2f" % acc )
        return acc
class fineTuningWiAR(fineTuningWidar):
    def __init__( self,config,idx_user):
        super( ).__init__( config = config, isMultiDomain = False,isiheritance = True )
        self.config = config
        self.wiar = WiARdataLoader( config, data_path = f'E:\\Sensing_project\\Cross_dataset\\WiAR\\volunteer_{idx_user}' )
        self.data = self.wiar.data
        self.label = self.wiar.label
        self.config.N_novel_classes = len(self.data)
        self.nways = self.config.N_novel_classes
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        self.pretrained_featureExtractor.trainable = True
        self.fine_Tune_model = self.modelObj.buildTuneModel(
                pretrained_feature_extractor = self.pretrained_featureExtractor,
                isTest = False, config = self.config
                )
    def tuning( self):
        self.pretrained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        self.pretrained_featureExtractor.trainable = True
        self.data = self.wiar.getSQDataForTest()
        weights = np.transpose(
                self._getNShotsEmbedding( self.pretrained_featureExtractor, self.data[ 'Support_data' ] )
                )
        bias = np.zeros( self.config.N_novel_classes )
        self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
        val_data, val_label = self.data[ 'Val_data' ], to_categorical(
                self.data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
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
                = self.config.N_novel_classes ),
                epochs = 1000,
                verbose = 0,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,self.data['record'],history]
def searchBestSample(nshots=None):
    config = getConfig( )
    data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
                 'E:/Sensing_project/Cross_dataset/20181109/User2',
                 'E:/Sensing_project/Cross_dataset/20181109/User3'
                 ]
    x = 3
    config.nshots_per_domain = None
    # config.nshots = int(5*1*1*config.nshots_per_domain)
    config.nshots = nshots
    config.train_dir = data_dir[0]
    # config.train_dir = 'E:/Cross_dataset/20181115'
    config.N_novel_classes = 6
    config.lr = 1e-4
    config.domain_selection = (2, 2, 3)
    # config.pretrainedfeatureExtractor_path = './models/signFi_featureExtractor_weight_AlexNet_lab_training_acc_0.95_on_250cls.h5'
    config.pretrainedfeatureExtractor_path = './a.h5'
    fineTuningWidarObj = fineTuningWidar(config = config,isMultiDomain = False)
    location,orientation,Rx = config.domain_selection
    val_acc = 0
    acc_record = []
    for i in range(100):
        fine_Tune_model,record,history = fineTuningWidarObj.tuning( init_bias = True)
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        acc_record.append(history.history[ 'val_acc' ][ -1 ])
        if val_acc < history.history['val_acc'][-1]:
            val_acc = history.history[ 'val_acc' ][ -1 ]
            config.tunedModel_path = f'./models/Publication_related/widar_fineTuned_model_20181109' \
                                     f'_{config.nshots}shots_' \
                                     f'_domain{config.domain_selection}_{val_acc:0.2f}_newFE_user{x}.h5'
            fine_Tune_model.save_weights(config.tunedModel_path)
            best_record = record
            config.record = best_record
            print(f'Updated record is: {best_record}')
            mdic = {'record':best_record,
                    'val_acc':val_acc}
            # config.setMatSavePath(f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots"
            #                       f"_domain_{config.domain_selection}_20181109.mat")
            config.setMatSavePath(
                    f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots"
                    f"_domain_{config.domain_selection}_20181109_newFE_user{x}.mat"
                    )
            savemat( config.matPath, mdic )
            if val_acc >= 0.95:
                print(f'reached expected val_acc {val_acc}')
                break
    '''Testing'''
    config.getSampleIdx( )
    test_acc,[y_true,y_pred],label_true = fineTuningWidarObj.test(applyFinetunedModel = True)
    plt_cf = pltConfusionMatrix( )
    title = f'{config.nshots}_shot_sRx_{Rx}_domain_{config.domain_selection}'
    plt_cf.pltCFMatrix( y = label_true, y_pred = y_pred, figsize = (18,15),title = title )
    print(f'The average accuracy is {np.mean(acc_record)}')
    plt.savefig(f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot learning/Results/results_figs'
                f'/{config.nshots}shots_'
                f'{config.domain_selection}_finetuned_{test_acc[0]:0.2f}_user{x}.pdf')
    return test_acc
def searchBestSampleMultiRx(nshots:int = None,Rx:list = None):
    config = getConfig( )
    data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
                 'E:/Sensing_project/Cross_dataset/20181109/User2',
                 'E:/Sensing_project/Cross_dataset/20181109/User3'
                 ]
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    config.nshots = nshots
    config.train_dir = data_dir[ 0 ]
    config.N_novel_classes = 6
    config.lr = 1e-4
    config.domain_selection = Rx
    # selection = np.random.choice(config.domain_selection,n,replace = False)
    x = 1
    val_acc = 0.0
    fineTuningWidarObj = fineTuningWidar( config = config, isMultiDomain = True )
    for i in range(50):
        fine_Tune_model,record,history = fineTuningWidarObj.tuningMultiRx( )
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        if val_acc < np.max(history.history['val_acc']):
            val_acc = np.max(history.history['val_acc'])
            config.tunedModel_path = f'./models/Publication_related/widar_fineTuned_M_Rx_model_20181109' \
                                     f'_{config.nshots}shots_' \
                                     f'_domain{config.domain_selection}_{val_acc:0.2f}_newFE_user{x}.h5'
            fine_Tune_model.save_weights(config.tunedModel_path)
            config.record = record
            print(f'Updated record is: {config.record }')
            mdic = {'record':config.record ,
                    'val_acc':val_acc}
            config.setMatSavePath(
                    f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots"
                    f"_domain_{config.domain_selection}_20181109_MultiRx_newFE_user{x}.mat"
                    )
            savemat( config.matPath, mdic )
            if val_acc >= 0.75:
                print(f'reached expected val_acc {val_acc}')
                break
    return val_acc
def evaluationMultiRx(N_Rx,N_shots):
    acc = {'user1':[],'user2':[],'user3':[] }
    config = getConfig( )
    data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
                 'E:/Sensing_project/Cross_dataset/20181109/User2',
                 'E:/Sensing_project/Cross_dataset/20181109/User3'
                 ]
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    path = './models/Publication_related/Rx_specific/'

    # config.tunedModel_path = ['./models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '1]_0.37_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '2]_0.61_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '3]_0.53_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '4]_0.39_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '5]_0.45_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '6]_0.42_newFE_user1.h5',]
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1]_0.37_newFE_user1.h5'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2]_0.43_newFE_user1'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2, 3]_0.42_newFE_user1.h5'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2, 3, 4]_0.41_newFE_user1.h5'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2, 3, 4, 5]_0.40_newFE_user1.h5'
    config.N_novel_classes = 6
    config.nshots = N_shots
    config.tunedModel_path = [
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[1]_newFE_user1.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[2]_newFE_user1.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[3]_newFE_user1.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[4]_newFE_user1.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[5]_newFE_user1.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[6]_newFE_user1.h5',]
    config.matPath = [ f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"1]_20181109_MultiRx_newFE_user1",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"2]_20181109_MultiRx_newFE_user1",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"3]_20181109_MultiRx_newFE_user1",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"4]_20181109_MultiRx_newFE_user1",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"5]_20181109_MultiRx_newFE_user1",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"6]_20181109_MultiRx_newFE_user1" ]
    config.record = []
    for i in range(len(config.matPath)):
        config.record.append(loadmat( config.matPath[i] + '.mat' )[ 'record' ])
    # for j in range(3):
    config.train_dir = data_dir[ 0 ]
    config.domain_selection = [1,2,3,4,5,6]
    testWidar = fineTuningWidar( config, True )
    # fineTuneModelEvalObj = WidarDataloader( config = config, isMultiDomain = True )
    # data = fineTuneModelEvalObj.getMultiDomainSQDataForTest(1,False)
    for i in range(10):
        # acc['user1'].append(testWidar.testMultiRx( applyFinetunedModel = True,useWeightMatrix = True,n_Rx = N_Rx ))
        acc[ 'user1' ].append( testWidar.testMultiRxSep( N_Rx = N_Rx ) )
    return acc
def evaluation( domain_selection,nshots ):
    # config.getFineTunedModelPath( )
    # location, orientation, Rx = config.domain_selection
    data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
                 'E:/Sensing_project/Cross_dataset/20181109/User2',
                 'E:/Sensing_project/Cross_dataset/20181109/User3'
                 ]
    config = getConfig( )
    config.domain_selection = domain_selection
    config.nshots = nshots
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    # config.setMatSavePath(
    #         f"./Sample_index/Publication_related/sample_index_record_for_2_shots_domain_(2, 2, 3)_20181109.mat"
    #         )
    config.matPath = [f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[1]_20181109_MultiRx_newFE_user1",
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[2]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[3]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[4]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[5]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[6]_20181109_MultiRx_newFE_user1"]
    # config.getSampleIdx( )
    config.train_dir = data_dir[0]
    # config.tunedModel_path = f'./models/fine_tuning_widar/widar_fineTuned_model_20181109_1shots_test_domain_(2, 2, 3).h5'
    config.tunedModel_path = './models/Publication_related/widar_fineTuned_model_20181109_5shots__domain(2, 2, ' \
                             '3)_0.97_newFE_user2.h5'
    config.record = loadmat(config.matPath)['record']
    config.domain_selection = domain_selection
    config.N_novel_classes = 6
    fineTuneModelEvalObj = fineTuningWidar( config = config, isMultiDomain = False )
    test_acc,[y_true,y_pred],label_true = fineTuneModelEvalObj.test(applyFinetunedModel =True)
    plt_cf = pltConfusionMatrix( )
    plt_cf.pltCFMatrix(
            y = label_true, y_pred = y_pred, figsize = (12, 10),title = ""
            )
    return test_acc
def compareDomain():
    config = getConfig( )
    config.nshots_per_domain = 2
    config.nshots = int( 5 * 1 * 1 * config.nshots_per_domain )
    config.train_dir = 'E:/Cross_dataset/20181109/User1'
    config.N_novel_classes = 6
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
def tuningSignFi():
    config = getConfig( )
    config.source = [1,2,5,4,3]
    config.nshots = 1
    config.N_novel_classes = 25
    config.N_base_classes = 125
    # config.lr = 4e-4
    # config.lr = 1e-3
    # user 1 - 0.7e-3, 2,3 - 0.65e-3, 4 -
    config.lr = 0.65e-3
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    config.tunedModel_path = f'./models/Publication_related/signFi_finetuned_model_{config.nshots}_shots_' \
                             f'{config.N_novel_classes}_ways_256_1280.h5'
    # config.pretrainedfeatureExtractor_path = \
    #     './models/pretrained_feature_extractors/signFi_featureExtractor_weight_AlexNet_lab_training_acc_0' \
    #     '.95_on_200cls.h5'
    config.pretrainedfeatureExtractor_path = 'a.h5'
    tuningSignFiObj = fineTuningSignFi(config,isZscore = False)
    fine_Tune_model = tuningSignFiObj.tuning(init_bias = False)
    return fine_Tune_model
def testingSignFi(path,mode,N_train_classes,environment:str):
    # config = getConfig( )
    # config.nshots = 1
    # config.train_dir = 'D:\Matlab\SignFi\Dataset'
    # config.source = 'lab'
    # all_path = os.listdir( f'./models/pretrained_feature_extractors/' )
    # for i, path in enumerate( all_path ):
    #     n = re.findall( r'\d+', all_path[ i ] )[ 2 ]
    #     if int( n ) == 200:
    #         config.N_base_classes = int( n )
    #         config.N_novel_classes = 276 - config.N_base_classes
    #         print( f'{n} in environment {config.source}' )
    #         config.pretrainedfeatureExtractor_path = './models/pretrained_feature_extractors/' + path
    # tuningSignFiObj = fineTuningSignFi( config, isZscore = False )
    # acc_all = tuningSignFiObj.test( nway = None, applyFinetunedModel = False )

    config = getConfig( )
    modelObj = models( )

    config.source = environment
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    config.N_base_classes = N_train_classes
    # config.lr = 3e-4
    config.pretrainedfeatureExtractor_path = path
    # Declare objects
    dataLoadObj = signDataLoader( config = config )
    # preTrain_modelObj = PreTrainModel( config = config )
    train_data, train_labels, test_data, test_labels = dataLoadObj.getFormatedData(
            source = config.source,
            isZscore = False
            )
    feature_extractor = modelObj.buildFeatureExtractor( mode = mode )
    feature_extractor.load_weights(config.pretrainedfeatureExtractor_path )
    fineTuningSignFiObj = fineTuningSignFi( config )
    test_acc = fineTuningSignFiObj.signTest(test_data, test_labels, 1000, feature_extractor)
    return test_acc
def tuningWiar(nshots,idx_user):
    config = getConfig()
    config.nshots = nshots
    config.lr = 1e-4
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    wiarFTObj = fineTuningWiAR(config = config,idx_user=idx_user)
    val_acc = 0.0
    for i in range(50):
        fine_Tune_model,record,history = wiarFTObj.tuning()
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        print('Tuning %d shots model' % nshots)
        currentAcc = np.max( history.history[ 'val_acc' ])
        print(f'Accuracy is {currentAcc}')
        if val_acc < np.max(history.history['val_acc']):
            val_acc = np.max(history.history['val_acc'])
            config.tunedModel_path = f'.\\models\\Publication_related\\wiar_FT_{config.nshots}shots_' \
                                     f'{val_acc:0.2f}_User_{idx_user}.h5'
            fine_Tune_model.save_weights(config.tunedModel_path)
            print(f'Updated record is: {record}')
            mdic = {'record':record,
                    'val_acc':val_acc}
            config.setMatSavePath(
                    f".\\Sample_index\\Publication_related\\sample_index_record_for_{config.nshots}_shots_User_{idx_user}.mat"
                    )
            savemat( config.matPath, mdic )
            # if val_acc >= 0.75:
            #     print(f'reached expected val_acc {val_acc}')
            #     break
    return val_acc
if __name__ == '__main__':
    '''WiAR'''
    acc_all = { }
    shots_list = [1,2,3,4,5]
    for nshots in shots_list:
        val_acc = tuningWiar(nshots = nshots,idx_user = 6 )
        acc_all[f'User1_{nshots}_shots'] = val_acc
    key = list(acc_all.keys())
    for i in range(len(key)):
        print(acc_all[key[i]])
    '''Multiple receivers test'''
    # acc_all = {}
    # for i in range(1,7):
    #     for shot in range(2,6):
    #         acc = evaluationMultiRx( N_Rx = i,N_shots=shot )
    #         acc_all[f'{i}_Rx_{shot}_shot'] = acc
    # savemat('multiRx_acc.mat',acc_all)
    # for k in range(1,7):
    #     x = 3
    #     per = np.min(acc_all[f'{k}_Rx_{x}_shot']['user1'])
    #     print(f'{k} receivers, {x} shots accuracy is: {per:.2f}%')

