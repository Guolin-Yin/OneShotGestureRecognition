from methodTesting.t_SNE import *
import os
import random
import numpy as np
import re
import copy
import math

import scipy.io as sio
from os.path import dirname, join
from tensorflow.keras.utils import to_categorical
from Preprocess.SignalPreprocess import *
import matplotlib.pyplot as plt
from scipy import stats
from Config import getConfig
from Preprocess.SignalPreprocess import *
from Preprocess.MMD import *
from scipy.io import savemat,loadmat
import hdf5storage
class WiARdataLoader:
    def __init__(self,config,data_path):
        self.config = config
        self.data_path = data_path
        self.filename = os.listdir( data_path )
        self.data, self.label = self._loaddataNlabels( )
    def _loaddataNlabels( self ):
        data = []
        label = []
        for count, currentPath in enumerate(self.filename):
            currentPath = os.path.join(self.data_path,currentPath)
            data_amp = sio.loadmat( currentPath )[ 'csiAmplitude' ]
            data_phase = sio.loadmat( currentPath )[ 'csiPhase' ]
            data.append( np.concatenate( (data_amp, data_phase), axis = 1 ) )
            label.append(int( re.findall( r'\d+', self.filename[count] )[0]))
        data, label = np.asarray( data ), np.asarray( label )
        classes = np.unique( label )
        cls = { }
        out_label = []
        for i in classes:
            idx = np.where( label == i )[ 0 ]
            cls[ f'act_{i-1}'] = data[idx]
            out_label.append(label[idx] - 1)

        return cls, np.concatenate( out_label )
    def getSQDataForTest( self ):

        gesture_type = list( self.data.keys( ) )
        num_sample_per_gesture = len(self.data[gesture_type[0]])
        num_val = num_sample_per_gesture - self.config.nshots
        support_set = [ ]
        query_set = [ ]
        support_label = [ ]
        query_label = [ ]
        record = [ ]
        Val_set = np.zeros( (len(gesture_type) * num_val, 200, 60, 3))
        Val_set_label = [ ]
        for count, gesture in enumerate( gesture_type ):
            idx_list = np.arange( 0, num_sample_per_gesture )
            shots_idx = np.random.choice( idx_list, self.config.nshots, replace = False )
            for i in shots_idx:
                idx_list = np.delete( idx_list, np.where( idx_list == i ) )
                support_set.append( self.data[ gesture ][ i ] )
                support_label.append( count )
            sample_idx = np.random.choice( idx_list, 1, replace = False )[ 0 ]
            query_set.append( self.data[ gesture ][ sample_idx ] )
            query_label.append( count )
            Val_set[ count * num_val:count * num_val + num_val, :, :, : ] = self.data[ gesture ][ idx_list ]
            [ Val_set_label.append( count ) for i in range( num_val ) ]
            record.append( shots_idx )
        Support_data = np.asarray( support_set )
        Support_label = np.expand_dims( np.asarray( support_label ), axis = 1 )
        Query_data = np.asarray( query_set )
        Query_label = np.expand_dims( np.asarray( query_label ), axis = 1 )
        Val_data = Val_set
        Val_label = np.expand_dims( (Val_set_label), axis = 1 )

        output = {
                'Support_data' : Support_data,
                'Support_label': Support_label,
                'Query_data'   : Query_data,
                'Query_label'  : Query_label,
                'Val_data'     : Val_data,
                'Val_label'    : Val_label,
                'record'       :record
                }
        return output
class gestureDataLoader:
    def __init__(self,batch_size :int = 32,data_path:str = 'D:/OneShotGestureRecognition/20181115/'):
        self.data_path = data_path
        self.batch_size = batch_size
        self.filename = os.listdir( data_path )
        self.csiAmplitude = np.array( [ ] )
        self.labels = np.asarray( [ ] )
        self.gesture_class = {}
        self.preprocessers = Denoiser( )
        x = []
        for name in self.filename:
            x.append(int(re.findall( r'\d+\b', name )[1]))
        self.num_gesture_types = np.max(x)
        self._mapFilenameToClass( )
        self._getInputShape( )
    def _getInputShape(self):
        data = sio.loadmat( os.path.join( self.data_path, self.filename[ 0 ] ) )[ 'csiAmplitude' ]
        self.InputShape = list(data.shape)
        self.num_subcarriers = self.InputShape[0]
        self.len_signals = self.InputShape[1]
    # def load( self, isTrain: bool, user:str ):
    #     buf = []
    #     for i in range(len(self.filename)):
    #         if (self.filename[i][4] == user):
    #             mat_fname = join(self.data_dir, self.filename[i])
    #             data = sio.loadmat(mat_fname)['csiAmplitude']
    #             buf.append(data)
    #             print(f'loadding     ({(i/len(self.filename))*100.0:.2f}%)')
    #     self.csiAmplitude = np.asarray(buf)
    # def getSelectedDomain(self,selected_location:int,selected_direction:int,selected_receiver:int):
    #     '''
    #     id-a-b-c-d-Rx.dat
    #     'id': user's id;
    #     'a': gesture type,
    #     'b': torso location,
    #     'c': face orientation,
    #     'd': repetition number,
    #     'Rx': Wi-Fi receiver id.
    #     :return: one hot labels, selected dataset
    #     '''
    #     '''
    #     1: Push&Pull;
    #     2: Sweep;
    #     3: Clap;
    #     4: Slide;
    #     5: Draw-O(Horizontal);
    #     6: Draw-Zigzag(Horizontal);
    #     7:Draw-N(Horizontal);
    #     8: Draw-Triangle(Horizontal);
    #     9: Draw-Rectangle(Horizontal);
    #     '''
    #     idx = []
    #     labels = []
    #     for i in range(len(self.filename)):
    #         gestureType = int(self.filename[i][6]) - 1
    #         location = int(self.filename[i][8])
    #         direction = int(self.filename[i][10])
    #         rx = int(self.filename[i][-5])
    #
    #         if location == selected_location:
    #             if direction == selected_direction:
    #                 if rx == selected_receiver:
    #                     idx.append(i)
    #                     labels.append(to_categorical(gestureType, N_base_classes = 9))
    #                     print(self.filename[i])
    #     selectedData = self.csiAmplitude[idx,:,:]
    #     return selectedData,labels
    def _preprocessData(self,data):
        x = standardisers( data )
        y = remove_zero( x )
        z = get_deonised_data( y, cf_freq=100 )
        e = get_median_dnData( data=z, size=7, mode="array" )
        # f, _ = pca_denoise( data=e, n_comp=7 )
        return e
    def _mapFilenameToClass(self):
        date_filename = re.findall( r'\b\d+\b', self.data_path )[0]
        sim_gesture_6ges = ['20181115','20181109','Pre_16','20181121','20181211','20181127']
        draw_gesture_10ges = [ '20181112', '20181116','Pre_16' ]
        if date_filename in sim_gesture_6ges:
            keys = [ 'Push&Pull',
                     'Sweep',
                     'Clap',
                     'Draw-O(Vertical)',
                     'Draw-Zigzag(Vertical)',
                     'Draw-N(Vertical)']
            for g_type in keys:
                recordGesture = [ ]
                for currentFileName in self.filename:
                    if int( re.findall(r'\d+\b',currentFileName)[ 1 ] ) == int( keys.index( g_type ) ) + 1:
                        filePath = os.path.join( self.data_path, currentFileName )
                        recordGesture.append( filePath )
                self.gesture_class[ g_type ] = recordGesture
        if date_filename in draw_gesture_10ges:
            keys = [("Draw-"+ str(i)) for i in range(1,self.num_gesture_types+1)]
            for g_type in keys:
                recordGesture = []
                for currentFileName in self.filename:
                    if int(re.findall(r'\d+\b',currentFileName)[ 1 ]) == int(keys.index(g_type)) + 1:
                        filePath = os.path.join(self.data_path,currentFileName)
                        recordGesture.append(filePath)
                self.gesture_class[g_type] = recordGesture
    def _mapPathToDataAndLabels(self,path:list,is_triplet_loss:bool,\
                                is_one_shot_task:bool=None,nshots:int=None):
        if not is_triplet_loss:
            num_of_pairs = int(len(path)/2)
            pairs_of_samples = [ np.zeros( (num_of_pairs, self.num_subcarriers, self.len_signals) ) for i in range( 2 ) ]
            labels = np.zeros((num_of_pairs,1))
            for pair in range(num_of_pairs):
                data = sio.loadmat(path[pair * 2])['csiAmplitude']
                pairs_of_samples[ 0 ][ pair, :, : ,0] = self._preprocessData(data)
                data = sio.loadmat( path[ pair * 2 + 1 ] )[ 'csiAmplitude' ]
                pairs_of_samples[ 1 ][ pair, :, : ,0] = self._preprocessData(data)
                # labels
                if not is_one_shot_task:
                    if pair % 2 == 0:
                        labels[pair] = 1
                    else:
                        labels[pair] = 0
            return pairs_of_samples, labels
        if is_triplet_loss:
            if not (nshots == None):
                num_pairs = int( len( path ) / (nshots + 1) )
                triplets = [ np.zeros( (self.batch_size, nshots, self.num_subcarriers, self.len_signals) ) ,\
                             np.zeros( (self.batch_size, self.num_subcarriers, self.len_signals) )]
                for i in range( num_pairs ):
                    data = []
                    for n in range(nshots + 1):
                        data.append( sio.loadmat( path[ i * (nshots + 1) + n ] )[ 'csiAmplitude' ] )
                    triplets[ 0 ][i,:nshots,:,:] = np.asarray(data[0:nshots])
                    triplets[ 1 ][i,:,:] = np.asarray(data[nshots:nshots+1])
            else:
                num_triplets = int(len(path)/3)
                triplets = [ np.zeros( (self.batch_size, self.num_subcarriers, self.len_signals) ) for i in range( 3 ) ]
                for i in range(num_triplets):
                    data = sio.loadmat( path[ i * 3 ] )[ 'csiAmplitude' ]
                    triplets[ 0 ][ i, :, : ] = data
                    data = sio.loadmat( path[ i * 3 + 1 ] )[ 'csiAmplitude' ]
                    triplets[ 1 ][ i, :, :  ] = data
                    data = sio.loadmat( path[ i * 3 + 2 ] )[ 'csiAmplitude' ]
                    triplets[ 2 ][ i, :, : ] = data
            return triplets
    def getTrainBatcher(self):
        '''
        Gesture type: 6
        '''
        Num_Gesture_Types = len(self.gesture_class)
        selectedGestureTypeIdx = [random.randint(0,Num_Gesture_Types - 1 ) for i in range(self.batch_size)]
        batch_gesture_path = []
        for index in selectedGestureTypeIdx:
            gestureType = list(self.gesture_class.keys())[index]
            All_available_current_Gesture_path = self.gesture_class[gestureType]
            num_geture_samples = len(All_available_current_Gesture_path)
            gesture_sample_index = random.sample( range( 0, num_geture_samples ), 3 )
            # Positive pair
            batch_gesture_path.append( All_available_current_Gesture_path[ gesture_sample_index[ 0 ] ] )
            batch_gesture_path.append( All_available_current_Gesture_path[ gesture_sample_index[ 1 ] ] )
            # Negative pair
            batch_gesture_path.append( All_available_current_Gesture_path[ gesture_sample_index[ 2 ] ] )
            different_gesture = copy.deepcopy(self.gesture_class)
            different_gesture.pop(gestureType)
            different_gesture_index = random.sample(range(0,len(different_gesture)),1)[0]
            gestureType = list( different_gesture.keys( ) )[ different_gesture_index ]
            All_available_current_Gesture_path = different_gesture[gestureType]
            num_geture_samples = len( All_available_current_Gesture_path )
            different_gesture_sample_index = random.sample(range(0,num_geture_samples),1)
            batch_gesture_path.append(All_available_current_Gesture_path[different_gesture_sample_index[0]])
        data, labels = self._mapPathToDataAndLabels( batch_gesture_path, is_triplet_loss=False, is_one_shot_task=False )
        return data,labels
    def getTripletTrainBatcher(self,isOneShotTask:bool = False,nShots:int = None):
        '''
        prepare the triplet batches for training.
        Every sample of our batch will contain 3 pictures :
        The Anchor, a Positive and a Negative.
        :return: triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
        https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
        '''
        # initialize result
        triplets_path = []
        for index in range( self.batch_size ):
            # Pick one random class for anchor
            rand_gesture_idx = random.randint( 0, self.num_gesture_types - 1 )
            anchor_gesture_Type = list( self.gesture_class.keys( ) )[rand_gesture_idx ]
            All_available_current_Gesture_path = self.gesture_class[ anchor_gesture_Type ]
            if isOneShotTask:
                # Pick two different random samples for this class => Anchor and Positive
                [ idx_Anchor, idx_Positive ] = np.random.choice( len(All_available_current_Gesture_path), size=2, replace=False )
                triplets_path.append( All_available_current_Gesture_path[ idx_Anchor ] )
                triplets_path.append( All_available_current_Gesture_path[ idx_Positive ] )

                # Pick another class for Negative, different from anchor_gesture
                different_gesture_type = copy.deepcopy( self.gesture_class )
                different_gesture_type.pop( anchor_gesture_Type )
                # rand_negative_gesture_type_idx = random.randint( 0, self.num_gesture_types - 1 )
                rand_negative_gesture_type_idx = random.randint( 0, len(different_gesture_type) - 1 )
                negative_gesture_type= list( different_gesture_type.keys( ) )[rand_negative_gesture_type_idx ]

                All_available_current_Gesture_path = different_gesture_type[negative_gesture_type]
                idx_Negative = np.random.choice(len(All_available_current_Gesture_path),size=1,replace=False)
                triplets_path.append( All_available_current_Gesture_path[ idx_Negative[0] ] )
            if not isOneShotTask:
                idx = np.random.choice( len( All_available_current_Gesture_path ), size=nShots+1, replace=False )
                for s in idx:
                    triplets_path.append( All_available_current_Gesture_path[ s ] )
        if isOneShotTask:
            data = self._mapPathToDataAndLabels( triplets_path, is_triplet_loss=True)
        if not isOneShotTask:
            data = self._mapPathToDataAndLabels( triplets_path, is_triplet_loss=True, nshots=nShots )

        return data
    def tripletsDataGenerator(self):
        while True:

            a,p,n = self.getTripletTrainBatcher( )
            labels = np.ones(self.batch_size)
            yield [a,p,n], labels
    def DirectLoadData(self, dataDir ):
        print( 'Loading data.....................................' )

        data = [ ]
        labels = [ ]
        gesture_6 = [ 'E:/Widar_dataset_matfiles/20181109/User1',
                      'E:/Widar_dataset_matfiles/20181109/User2', ]
        gesture_10 = [ 'E:/Widar_dataset_matfiles/20181112/User1',
                       'E:/Widar_dataset_matfiles/20181112/User2',
                       'Combined_link_dataset/20181116' ]
        for Dir in dataDir:
            fileName = os.listdir( Dir )
            for name in fileName:
                if re.findall( r'\d+\b', name )[ 5 ] == '3':
                    print( f'Loading {name}' )
                    path = os.path.join( Dir, name )
                    data.append( preprocessData( sio.loadmat( path )[ 'csiAmplitude' ] ) )
                    if Dir in gesture_6:
                        gestureMark = int( re.findall( r'\d+\b', name )[ 1 ] ) - 1
                    elif Dir in gesture_10:
                        gestureMark = int( re.findall( r'\d+\b', name )[ 1 ] ) + 6 - 1
                    labels.append( tf.keras.utils.to_categorical( gestureMark, num_classes=config.N_base_classes ) )
        return np.asarray( data ), np.asarray( labels )
class WidarDataloader(gestureDataLoader):
    def __init__(self,isMultiDomain:bool = False,config=None):
        super().__init__(data_path = config.train_dir)
        self.config = config
        self.selection = config.domain_selection
        self.preprocessers = Denoiser( )
        if isMultiDomain:
            print(f'Using the data from domain: {self.selection}')
            self.multiRx_data = self._getMultiOrientationData( self.selection )
        else:
            self.selected_gesture_samples_data = self._mapClassToDataNLabels(
                    selected_gesture_samples_path = self._selectPositions( selection = self.selection )
                    )
    def _getMultiOrientationData(self,selection):
        # _, _, Rx = selection
        selected_multiorientation_gesture_samples_data = {}

        for location in [2]:
            for orientation in [2]:
                for receiver in selection:
                    domain = (location,orientation,receiver)
                    path = self._selectPositions( domain )
                    data = self._mapClassToDataNLabels(path)
                    selected_multiorientation_gesture_samples_data[ f'{domain}' ] = data
        return selected_multiorientation_gesture_samples_data
    def _selectPositions(self,selection : tuple):
        location, orientation, Rx = selection
        selected_gesture_samples_path = {}
        for currentGesture in self.gesture_class:
            all_path = self.gesture_class[currentGesture]
            selected_path = []
            for currentFileName in all_path:
                '''
                0: date
                1: user ID
                2: gesture type
                3: location
                4: orientation
                5: repetition
                6: Rx ID
                '''
                # location
                if int( re.findall(r'\d+\b',currentFileName)[ -4 ] ) == location:
                    if int( re.findall( r'\d+\b', currentFileName )[-3 ] ) == orientation:
                        if int( re.findall( r'\d+\b', currentFileName )[ -1 ] ) == Rx:
                            selected_path.append(currentFileName)
            selected_gesture_samples_path[ currentGesture ] = selected_path
        return selected_gesture_samples_path
    def _getZscoreData( self, x):
        x = stats.zscore( x, axis = 0, ddof = 0 )
        return x
    def _mapClassToDataNLabels( self,selected_gesture_samples_path, ampOnly=False ):
        def sanitisePhases(phase):
            out_phase = np.zeros((phase.shape))
            for i in range(len(phase)):
                for ant in range(phase.shape[2]):
                    out_phase[i,:,ant] = self.preprocessers.phaseSanitizer(phase[i,:,ant])
            return out_phase
        gesture = {}
        x_all = []
        y_all = []
        for currentGesture in selected_gesture_samples_path:
            all_path = selected_gesture_samples_path[ currentGesture ]
            data = []
            # labels = []
            for currentPath in all_path:
                data_amp = sio.loadmat( currentPath )[ 'csiAmplitude' ]
                # data_phase = sanitisePhases(sio.loadmat(currentPath)['csiPhase'])
                data_phase = sio.loadmat( currentPath )[ 'csiPhase' ]
                # data_amp, data_phase = self.preprocessers.csiRatio( isWidar = True, data_amp = data_amp, data_phase = \
                #     data_phase)
                if ampOnly:
                    real = data_amp * np.cos( data_phase )
                    imag = data_amp * np.sin( data_phase )
                    complex = real + 1j * imag
                    data.append(complex)
                    x_all.append(complex)
                else:
                    # data_amp = self._getZscoreData( data_amp )
                    # data_phase = self._getZscoreData( data_phase )
                    data.append(np.concatenate( (data_amp, data_phase), axis = 1 ))
                    x_all.append(np.concatenate( (data_amp, data_phase), axis = 1 ))
                # labels.append(int( re.findall(r'\d+\b',currentPath)[ 2 ] ) - 1)
                y_all.append(int( re.findall(r'\d+\b',currentPath)[ 2 ] ) - 1)
            gesture[currentGesture] = np.asarray(data)
            self.gesture_type = list( gesture.keys( ) )
        return gesture
    def getSQDataForTest( self,nshots: int,mode:str, isTest:bool=False,Best = None):
        gesture_type = list( self.selected_gesture_samples_data.keys( ) )
        support_set = [ ]
        query_set = [ ]
        support_label = [ ]
        query_label = [ ]
        num_sample_per_gesture = 20
        num_val = num_sample_per_gesture-nshots
        Val_set = np.zeros( (6 * num_val, 200, 60, 3) )
        Val_set_label = [ ]
        record = []
        if mode == 'random':
            for gesture in gesture_type:
                sample_idx = np.random.choice( np.arange( 0, num_sample_per_gesture ), nshots+1, replace = False )
                [support_set.append( self.selected_gesture_samples_data[gesture][sample_idx[i]] ) for i in range(nshots)]
                query_set.append( self.selected_gesture_samples_data[gesture][sample_idx[-1]] )
            return np.asarray(support_set),np.asarray(query_set)
        if mode == 'fix':
            for count, gesture in enumerate(gesture_type):
                if not isTest:
                    idx_list = np.arange( 0, num_sample_per_gesture )
                    shots_idx = np.random.choice( idx_list, nshots, replace = False )
                    for i in shots_idx:
                        idx_list = np.delete( idx_list, np.where( idx_list == i ) )
                        support_set.append( self.selected_gesture_samples_data[ gesture ][ i ] )
                        support_label.append(count)
                    sample_idx = np.random.choice( idx_list, 1, replace = False )[0]
                    query_set.append( self.selected_gesture_samples_data[ gesture ][ sample_idx ] )
                    query_label.append(count)
                    Val_set[count*num_val:count*num_val+num_val,:,:,:] = self.selected_gesture_samples_data[ gesture ][idx_list]
                    [Val_set_label.append(count) for i in range(num_val)]
                    record.append(shots_idx )
                else:
                    idx_list = np.arange( 0, num_sample_per_gesture )
                    shots_idx =  Best[ count ]
                    for i in shots_idx:
                        idx_list = np.delete( idx_list, np.where( idx_list == i ) )
                        support_set.append( self.selected_gesture_samples_data[ gesture ][ i ] )
                        support_label.append( count )
                    sample_idx = np.random.choice( idx_list, 1, replace = False )[ 0 ]
                    query_set.append( self.selected_gesture_samples_data[ gesture ][ sample_idx ] )
                    query_label.append( count )
                    Val_set[ count * num_val:count * num_val + num_val, :, :, : ] = self.selected_gesture_samples_data[ gesture ][ idx_list ]
                    [ Val_set_label.append( count ) for i in range( num_val ) ]
                    record.append( shots_idx )

            Support_data = np.asarray(support_set)
            Support_label = np.expand_dims(np.asarray(support_label),axis=1)
            Query_data = np.asarray(query_set)
            Query_label = np.expand_dims(np.asarray(query_label),axis=1)
            Val_data = Val_set
            Val_label = np.expand_dims((Val_set_label),axis=1)

            output = {  'Support_data':Support_data,
                        'Support_label':Support_label,
                        'Query_data':Query_data,
                        'Query_label':Query_label,
                        'Val_data':Val_data,
                        'Val_label':Val_label,
                        'record':record
                    }
            return output
        if mode == 'multiRx':
            for count, gesture in enumerate( gesture_type ):
                if not isTest:
                    idx_list = np.arange( 0, num_sample_per_gesture )
                    shots_idx = np.random.choice( idx_list, nshots, replace = False )
                    for i in shots_idx:
                        idx_list = np.delete( idx_list, np.where( idx_list == i ) )
                        support_set.append( self.selected_gesture_samples_data[ gesture ][ i ] )
                        support_label.append(count)
                    sample_idx = np.random.choice( idx_list, 1, replace = False )[0]
                    query_set.append( self.selected_gesture_samples_data[ gesture ][ sample_idx ] )
                    query_label.append(count)
                    Val_set[count*num_val:count*num_val+num_val,:,:,:] = self.selected_gesture_samples_data[ gesture ][idx_list]
                    [Val_set_label.append(count) for i in range(num_val)]
                    record.append(shots_idx )
    def _delete_idx(self, idx_list,shots_idx, nshots_per_domain):
        for n in range( nshots_per_domain ):
            idx_list = np.delete( idx_list, list( idx_list ).index( list( shots_idx )[ n ] ) )
        return idx_list
    def getMultiDomainSQDataForTest( self,nshots_per_domain,isTest:bool,Best = None ):

        gesture_type = self.gesture_type
        # Support_set = { }
        # query_set = { }
        # Support_set = dict.fromkeys( gesture_type, [ ] )
        # query_set = dict.fromkeys( gesture_type, [ ] )
        Support_set = {  'Push&Pull': [],
                         'Sweep': [],
                         'Clap': [],
                         'Draw-O(Vertical)': [],
                         'Draw-Zigzag(Vertical)': [],
                         'Draw-N(Vertical)': []}
        query_set = {  'Push&Pull': [],
                         'Sweep': [],
                         'Clap': [],
                         'Draw-O(Vertical)': [],
                         'Draw-Zigzag(Vertical)': [],
                         'Draw-N(Vertical)': []}
        support_label = [ ]
        query_label = [ ]
        n_samples_perCls = 10
        num_val = n_samples_perCls - nshots_per_domain
        Val_set_multi_domain = []
        Val_set_label_multi_domain = [ ]
        record = []
        multiRx_data = self.multiRx_data

        for count, gesture in enumerate( gesture_type ):
            all_domain = list( multiRx_data )
            if not isTest:
                Current_record = [ ]
                for i in range( len( multiRx_data ) ):
                    idx_list = np.arange( 0, n_samples_perCls )
                    ########
                    shots_idx = np.random.choice( idx_list, nshots_per_domain, replace = False )
                    #########
                    # randIdx = np.random.choice( np.arange( 0, len(all_domain) ), 1, replace = False )[0]
                    randIdx = 0
                    current_domain = all_domain[ randIdx ]
                    all_domain.pop(randIdx)
                    idx_list = self._delete_idx(idx_list,shots_idx,nshots_per_domain)
                    # Support_set.append(
                    #         multiRx_data[ current_domain ][ gesture ][
                    #             shots_idx ]
                    #         )
                    Support_set[ gesture ].append(multiRx_data[ current_domain ][gesture ][shots_idx ])
                    # sample_idx = np.random.choice( idx_list, num_val, replace = False )[ 0 ]
                    query_set[ gesture ].append(multiRx_data[ current_domain ][ gesture ][idx_list ])
                    [support_label.append( count ) for n in range( nshots_per_domain )]
                    [ query_label.append( count ) for n in range( len(idx_list) ) ]
                    Val_set_multi_domain.append(multiRx_data[ current_domain ][ gesture ][idx_list ])
                    [ Val_set_label_multi_domain.append( count ) for m in range( len(idx_list) ) ]
                    Current_record.append( shots_idx )
                record.append(np.asarray(Current_record).reshape(len( multiRx_data ), nshots_per_domain ) )
                # all_domain = list( multiRx_data )
                # domain_idx = np.random.choice( np.arange( 0, len( all_domain ) ), 1, replace = False )[ 0 ]
                # selected_domain = all_domain[ domain_idx ]
                # sample_idx = np.random.choice( idx_list, 1, replace = False )[ 0 ]
                # query_set.append(
                #         multiRx_data[ selected_domain ][
                #             gesture ][ sample_idx ]
                #         )
            else:
                # idx_list = np.arange( 0, 20 )
                # current_domain_shots_idx = Best[ count ]

                for i in range( len( multiRx_data ) ):
                    idx_list = np.arange( 0, n_samples_perCls )
                    ################
                    record = Best
                    shots_idx = record[count][i][0]
                    #############
                    randIdx = 0
                    current_domain = all_domain[ randIdx ]
                    all_domain.pop( randIdx )
                    idx_list = self._delete_idx( idx_list, shots_idx, nshots_per_domain )
                    Support_set[ gesture ].append( multiRx_data[ current_domain ][ gesture ][ shots_idx ] )
                    # sample_idx = np.random.choice( idx_list, num_val, replace = False )[ 0 ]
                    query_set[ gesture ].append( multiRx_data[ current_domain ][ gesture ][ idx_list ] )
                    [ support_label.append( count ) for n in range( nshots_per_domain ) ]
                    [ query_label.append( count ) for n in range( len( idx_list ) ) ]
                    Val_set_multi_domain.append( multiRx_data[ current_domain ][ gesture ][ idx_list ] )
                    [ Val_set_label_multi_domain.append( count ) for m in range( len( idx_list ) ) ]
                #     idx_list = np.arange( 0, 20 )
                #     current_domain = all_domain[ i ]
                #     shots_idx = current_domain_shots_idx[ i, : ]
                #     # idx_list = np.delete( idx_list, np.where( idx_list == shots_idx ) )
                #     idx_list = self._delete_idx( idx_list, shots_idx, nshots_per_domain )
                #     Support_set.append(
                #             multiRx_data[ current_domain ][ gesture ][
                #                 shots_idx ]
                #             )
                #     support_label.append( count )
                #     Val_set_multi_domain.append(
                #             multiRx_data[ current_domain ][ gesture ][
                #                 idx_list ]
                #             )
                #     [ Val_set_label_multi_domain.append( count ) for i in range( num_val ) ]
                # domain_idx = np.random.choice( np.arange( 0, len( all_domain ) ), 1, replace = False )[ 0 ]
                # selected_domain = all_domain[ domain_idx ]
                # sample_idx = np.random.choice( idx_list, 1, replace = False )[ 0 ]
                # query_set.append(
                #         multiRx_data[ selected_domain ][
                #             gesture ][ sample_idx ]
                #         )
                # query_label.append( count )
        else:
            # Support_set = np.concatenate( Support_set, axis = 0 )
            # Support_data = np.asarray( Support_set )
            Support_label = np.expand_dims( np.asarray( support_label ), axis = 1 )
            # Query_data = np.asarray( query_set )
            Query_label = np.expand_dims( np.asarray( query_label ), axis = 1 )
            Val_data = np.concatenate(Val_set_multi_domain,axis = 0 )
            Val_label = np.expand_dims( (Val_set_label_multi_domain), axis = 1 )
            record = record
        output = {
                'Support_data' : Support_set,
                'Support_label': Support_label,
                'Query_data'   : query_set,
                'Query_label'  : Query_label,
                'Val_data'     : Val_data,
                'Val_label'    : Val_label,
                'record'       : record
                }
        return output
class signDataLoader:
    ''':returns
        filename: [0] home-276 -> user 5, 2760 samples,csid_home and csiu_home
        filename: [1] lab-150 -> user 1 to 5, 1500 samples/user
        filename: [2] lab-276 -> user 5, 5520 samples,downlink*
        filename: [3] lab-276 -> user 5, 5520 samples,uplink*
    '''
    def __init__( self,dataDir = None,config = None ):
        self.config = config
        self.dataDir = config.train_dir
        self.data = []
        self.data, self.filename = self.loadData()
        self.preprocessers = Denoiser( )
    def _reformat( self,ori_data ):
        reformatData = np.zeros((ori_data.shape[3],ori_data.shape[0],ori_data.shape[1],ori_data.shape[2]),dtype='complex_')
        for i in range(ori_data.shape[-1]):
            reformatData[i,:,:,:] = ori_data[:,:,:,i]
        return reformatData
    def loadData( self,  ):
        print("Loading data................")
        fileName = os.listdir( self.dataDir )
        for name in fileName:
            path = os.path.join(self.dataDir,name)
            buf = sio.loadmat(path)
            buf.pop( '__header__', None )
            buf.pop('__version__',None)
            buf.pop( '__globals__', None )
            for i in range(len(buf)):
                if 'label' in list( buf.keys( ) )[ i ]:
                    continue
                buf[list( buf.keys( ) )[ i ]] = self._reformat(buf[list( buf.keys( ) )[ i ]])
            self.data.append( buf )
        return [self.data,fileName]
    def _getConcatenated( self, x ,isZscore:bool):
        # x_amp,x_phase = self.preprocessers.csiRatio(csi = x)
        x_amp = np.abs( x )
        x_phase = np.angle( x )
        if isZscore:
            x_amp = stats.zscore( x_amp, axis = 1, ddof = 0 )
            x_phase = stats.zscore( x_phase, axis = 1, ddof = 0 )
        x_all = np.concatenate( (x_amp, x_phase), axis=2 )
        return x_all
    def getFormatedData(self,source:str='lab',isZscore:bool=False):
        def getSplitData(x_all,y_all,n_samples_per_user:int,shuffle=True):
            n_base_classes = self.config.N_base_classes
            n_test_classes = 276 - n_base_classes
            n_train_samples = n_base_classes * n_samples_per_user
            n_test_samples = (276 - n_base_classes) * n_samples_per_user
            train_data = np.zeros( (n_train_samples, 200, 60, 3) )
            train_labels = np.zeros( (n_train_samples, 1) ,dtype = int)
            test_data = np.zeros( (n_test_samples, 200, 60, 3) )
            test_labels = np.zeros( (n_test_samples, 1) ,dtype = int)
            idx = np.where( y_all == 1 )[0]
            tra_count = 0
            tes_count = 0
            for i in idx:
                train_data[tra_count:tra_count + n_base_classes, :, :, : ] = x_all[ i:i + n_base_classes, :, :, : ]
                train_labels[tra_count:tra_count + n_base_classes, : ] = y_all[ i:i + n_base_classes, : ]
                test_data[tes_count:tes_count+n_test_classes,:,:,:] = x_all[ i + n_base_classes:i + 276, :, :, : ]
                test_labels[tes_count:tes_count+n_test_classes,:] = y_all[ i + n_base_classes:i + 276, : ]
                tra_count += n_base_classes
                tes_count += n_test_classes
            if shuffle:
                idx = np.random.permutation( len( train_labels ) )
                train_data = train_data[idx]
                train_labels = train_labels[idx]
            return [train_data, train_labels, test_data, test_labels]
        if source == 'lab':
            print( 'lab environment user 5, 276 classes,5520 samples,downlink*' )
            x = self.data[ 2 ][ 'csid_lab' ]
            # x_amp,x_phase = self.preprocessers.csiRatio(csi = x)
            x_amp = np.abs( x )
            x_phase = np.angle( x )
            if isZscore:
                x_amp = stats.zscore( x_amp, axis = 1, ddof = 0 )
                x_phase = stats.zscore( x_phase, axis = 1, ddof = 0 )
            x_all = np.concatenate( (x_amp, x_phase), axis=2 )
            # x_all = x_phase
            y_all = self.data[ 2 ][ 'label_lab' ]
            train_data, train_labels, test_data, test_labels = getSplitData(x_all=x_all,y_all=y_all,
                    n_samples_per_user=20,shuffle=True)
            return [ train_data, train_labels, test_data, test_labels ]
        elif source == 'home':
            print('home environment user 5, 276 classes, 2760 samples')
            x = self.data[ 0 ][ 'csid_home' ]
            # x_amp,x_phase = self.preprocessers.csiRatio(csi = x)
            x_amp = np.abs( x )
            x_phase = np.angle( x )
            if isZscore:
                x_amp = stats.zscore( x_amp, axis = 1, ddof = 0 )
                x_phase = stats.zscore( x_phase, axis = 1, ddof = 0 )
            x_all = np.concatenate( (x_amp, x_phase), axis=2 )
            y_all = self.data[ 0 ][ 'label_home' ]
            train_data, train_labels, test_data, test_labels = getSplitData(
                    x_all = x_all, y_all = y_all,
                    n_samples_per_user = 10,shuffle=False
                    )
            return [ train_data, train_labels, test_data, test_labels ]
        elif type(source) == list:
            # if len( source ) < 4:
            #     sys.exit( 'list of items not long enough' )
            print(f'train on user {source[0]}-{source[1]}-{source[2]}-{source[3]}, test on user {source[4]}')
            source = [source[0]-1, source[1]-1, source[2]-1, source[3]-1, source[4]-1]
            x_1 = self._getConcatenated( self.data[ 1 ][ 'csi1' ], isZscore )
            x_2 = self._getConcatenated( self.data[ 1 ][ 'csi2' ], isZscore )
            x_3 = self._getConcatenated( self.data[ 1 ][ 'csi3' ], isZscore )
            x_4 = self._getConcatenated( self.data[ 1 ][ 'csi4' ], isZscore )
            x_5 = self._getConcatenated( self.data[ 1 ][ 'csi5' ], isZscore )
            y_1 = self.data[ 1 ][ 'label' ][ 0:1500 ]
            y_2 = self.data[ 1 ][ 'label' ][ 1500:3000 ]
            y_3 = self.data[ 1 ][ 'label' ][ 3000:4500 ]
            y_4 = self.data[ 1 ][ 'label' ][ 4500:6000 ]
            y_5 = self.data[ 1 ][ 'label' ][ 6000:7500 ]
            x = [x_1,x_2,x_3,x_4,x_5]
            y = [y_1,y_2,y_3,y_4,y_5]
            x_train = np.concatenate( (x[source[0]],x[source[1]],x[source[2]],x[source[3]]),axis = 0)
            y_train = np.concatenate(
                    (y[ source[ 0 ] ], y[ source[ 1 ] ], y[ source[ 2 ] ], y[ source[ 3 ] ]), axis = 0
                    )
            x_test = x[source[4]]
            y_test = y[source[4]]
            return [x_train,y_train,x_test,y_test]
    def getTrainTestSplit(self, data, labels, N_train_classes: int , N_samples_per_class: int ,
                           shuffle_training: bool = True ):
        if N_train_classes == 276:
            train_data = data
            train_labels = labels
            test_data = None
            test_labels = None
            return [ train_data, train_labels, test_data, test_labels ]
        N_samples = len( labels )
        N_classes = int( N_samples / N_samples_per_class )
        N_train_samples = N_train_classes * N_samples_per_class
        N_test_samples = N_samples - N_train_samples
        N_test_classes = int( N_test_samples / N_samples_per_class )

        train_data = np.zeros( (N_train_samples, 200, 60, 3) )
        train_labels = np.zeros( (N_train_samples, 1) )
        test_data = np.zeros( (N_test_samples, 200, 60, 3) )
        test_labels = np.zeros( (N_test_samples, 1) )
        count_tra = 0
        count_tes = 0
        for i in list( np.arange( 0, N_samples, N_classes ) ):
            train_data[ count_tra:count_tra + N_train_classes, :, :, : ] = data[ i:i + N_train_classes, :, :, : ]
            train_labels[ count_tra:count_tra + N_train_classes, : ] = labels[ i:i + N_train_classes, : ]
            test_data[ count_tes:count_tes + N_test_classes, :, :, : ] = data[ i + N_train_classes:i + N_classes, :, :,
                                                                         : ]
            test_labels[ count_tes:count_tes + N_test_classes, : ] = labels[ i + N_train_classes:i + N_classes, : ]
            count_tra += N_train_classes
            count_tes += N_test_classes
        if shuffle_training:
            idx = np.random.permutation( len( train_labels ) )
            train_data = train_data[ idx, :, :, : ]
            train_labels = train_labels[ idx, : ]
        return [ train_data, train_labels, test_data, test_labels ]
if __name__ == '__main__':
    config = getConfig( )
    config.source = [1,2,3,4,5]
    config.nshots = 1
    config.N_novel_classes = 25
    config.N_base_classes = 150 - config.N_novel_classes

    config.lr = 0.65e-3

    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    signData = signDataLoader(config = config)
    _,_,x_test,y_test = signData.getFormatedData( source = config.source)



    # config = getConfig( )
    # config.nshots = 1
    # wiar = WiARdataLoader(config,data_path = 'E:\\Sensing_project\\Cross_dataset\\WiAR\\volunteer_2')
    # data = wiar.getSQDataForTest()
    # data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
    #              'E:/Sensing_project/Cross_dataset/20181109/User2',
    #              'E:/Sensing_project/Cross_dataset/20181109/User3'
    #              ]
    # config = getConfig( )
    #
    # config.nshots = 1
    # config.pretrainedfeatureExtractor_path = './a.h5'
    # config.matPath = f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_(2, 2, " \
    #                  f"3)_20181109_newFE_user2.mat"
    # config.train_dir = data_dir[ 1 ]
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_model_20181109_5shots__domain(2, 2, ' \
    #                          '3)_0.97_newFE_user2.h5'
    # # config.record = loadmat( config.matPath )[ 'record' ]
    # # config.domain_selection = domain_selection
    # config.N_novel_classes = 6
    # config.domain_selection = [1,2,3,4,5]
    # fineTuneModelEvalObj = WidarDataloader( config = config, isMultiDomain = True,)
    # data = fineTuneModelEvalObj.getMultiDomainSQDataForTest(1,False,)
    #
    # Support_data = []
    # keys = list( data[ 'Support_data' ].keys( ) )
    # [ Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 ) ) for j in
    #   range( len( keys ) ) ]
    # s_data_array = np.concatenate(Support_data,axis=0)
    # a = loadmat('E:\BaiduNetdiskDownload\Data_part_1\gesture')
    # mat = hdf5storage.loadmat( filepath )
    # config = getConfig( )
    # preprocessers = Denoiser( )
    # WidarDataloader = WidarDataloader(dataDir = 'E:/Cross_dataset/20181109/User1', selection = (2,2,3))
    # signDataLoader = signDataLoader(dataDir = 'D:\Matlab\SignFi\Dataset')
    # data_lab = signDataLoader.getFormatedData(source = 'lab')
    # data_home = signDataLoader.getFormatedData( source = 'home' )
    # idx = np.where(data_home[3] == [251,252,253,254,255,256,257,258,259,260])[0]
    # class_t_sne(data_home[2][idx].reshape(100,-1),label = data_home[3][idx] - 250,perplexity = 7)
    # data_lab = data_lab[2].reshape(520,-1)
    # data_home = data_home[2].reshape(260,-1)
    '''========================================================================='''
    # config.train_dir = 'E:/Cross_dataset/20181109/User1'
    # WidarDataloaderObj = WidarDataloader(config.train_dir,selection = (2,2,3))
    # data = WidarDataloaderObj.selected_gesture_samples_data
    # keys = list(data.keys( ) )
    # data_all = []
    # for n in range(len(keys)):
    #     data_all.append(data[keys[n]])
    # data_all = np.concatenate((data_all),axis=0)
    # path = 'D:\OneShotGestureRecognition\Sample_index\sample_index_record_for_1_shots_domain_(2, 2, 3)_20181109.mat'
    # a = np.squeeze(loadmat(path)['record'])
    # label = []
    # for j in range(6):
    #     for i in range(20):
    #         if i == a[j]:
    #             label.append(j+10)
    #         else:
    #             label.append(j)
    # label = np.asarray(label)
    # class_t_sne(data_all.reshape(len(data_all),-1),label,perplexity=10,n_iter = 2000)
    '''========================================================================='''
    # domain_t_sne((data_lab[0:114],data_home[0:114],data_widar),perplexity = 6,n_iter = 3000)
    # config.domain_selection = (2,2,3)
    # path = 'E:/Cross_dataset/20181109/User1'
    # WidarDataloaderObj = WidarDataloader(dataDir = path,selection = config.domain_selection,config = config,isMultiDomain =
    # True)
    # output = WidarDataloaderObj.getMultiDomainSQDataForTest( nshots_per_domain = 3, isTest = False )
    # output_2 = WidarDataloaderObj.getMultiDomainSQDataForTest(
    #         nshots_per_domain = 3, isTest = True, Best = output[
    #             'record' ]
    #         )
    '''=====================================MMD===================================='''
    # signDataObj = signDataLoader(dataDir = 'D:\Matlab\SignFi\Dataset' )
    # a = signDataObj.getFormatedData(source = 'lab')
    # # data = []
    # # for i in range(len(a[0])):
    # #     data.append(a[0][i].reshape(-1))
    # # data = np.asarray(data)
    # data_1 = torch.tensor( a[ 0 ].reshape( len( a[ 0 ] ), -1 ) )
    # a = signDataObj.getFormatedData( source = 'home' )
    # data_2 = torch.tensor( a[ 0 ].reshape( len( a[ 0 ] ), -1 ) )
    # batch_size = 100
    # mmd_val = []
    # for i in np.arange(0,2500,100):
    #     mmd_val.append(mmd( data_1[i:i+batch_size], data_2[0:batch_size] ))
    # print( "MMD Loss:", np.mean(mmd_val) )
    #
    #
    # config.domain_selection = (2, 2, 3)
    # config.train_dir = 'E:/Cross_dataset/20181109/User1'
    # WidarDataLoaderObjMulti = WidarDataloader(
    #         dataDir = config.train_dir, selection = config.domain_selection, isMultiDomain = False,
    #         config = config
    #         )
    # data = WidarDataLoaderObjMulti.getSQDataForTest(
    #         nshots = 1, mode = 'fix',
    #         isTest = False, Best = None
    #         )
    # data_3 = torch.tensor(data['Val_data'].reshape(114,-1))
    # # print( "MMD Loss:", mmd( data_1[ 2*x:3*x ], data_3[ 0:x ] ) )
    # mmd_val = []
    # for i in np.arange(0,5000,100):
    #     mmd_val.append(mmd( data_1[i:i+batch_size], data_3 ))
    # print( "MMD Loss:", np.mean(mmd_val) )