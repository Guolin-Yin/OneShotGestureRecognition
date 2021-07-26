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
class gestureDataLoader:
    def __init__(self,batch_size :int = 32,data_path:str = 'D:/OneShotGestureRecognition/20181115/'):
        self.data_path = data_path
        self.batch_size = batch_size
        self.filename = os.listdir( data_path )
        self.csiAmplitude = np.array( [ ] )
        self.labels = np.asarray( [ ] )
        self.gesture_class = {}
        self._getInpuShape( )

        x = []
        for name in self.filename:
            x.append(int(re.findall( r'\d+\b', name )[1]))
        self.num_gesture_types = np.max(x)
        self._mapFilenameToClass( )
    def _getInpuShape(self):
        data = sio.loadmat( os.path.join(self.data_path,self.filename[0]) )[ 'csiAmplitude' ]
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
    #                     labels.append(to_categorical(gestureType, N_train_classes = 9))
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
        sim_gesture_6ges = ['20181115','20181109','Pre_16','20181121']
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
                    labels.append( tf.keras.utils.to_categorical( gestureMark, num_classes=config.N_train_classes ) )
        return np.asarray( data ), np.asarray( labels )
class signDataLoder:
    def __init__( self,dataDir ):
        self.dataDir = dataDir
        self.data = []
        self.data, self.filename = self.loadData()
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
    def getFormatedData(self,source:str='lab'):
        if source == 'lab':
            print( 'loading data from lab' )
            x = self.data[ 2 ][ 'csid_lab' ]
            x_amp = np.abs( x )
            x_phase = np.angle( x )
            x_all = np.concatenate( (x_amp, x_phase), axis=2 )
            y_all = self.data[ 2 ][ 'label_lab' ]
        elif source == 'home':
            print('loading data from home')
            x = self.data[ 0 ][ 'csid_home' ]
            x_amp = np.abs( x )
            x_phase = np.angle( x )
            x_all = np.concatenate( (x_amp, x_phase), axis=2 )
            y_all = self.data[ 0 ][ 'label_home' ]
        elif source == 'lab_other':
            x = self.data[ 1 ][ 'csi1' ][0:1500]
            x_amp = np.abs( x )
            x_phase = np.angle( x )
            x_all = np.concatenate( (x_amp, x_phase), axis=2 )
            y_all = self.data[ 1 ][ 'label' ][0:1500]
        return [x_all,y_all]

    def getTrainTestSplit(self, data, labels, N_train_classes: int = 260, N_samples_per_class: int = 20,
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
    signData = signDataLoder(dataDir = 'D:/Matlab/SignFi/Dataset')
    data,fileName = signData.loadData()
    lab_data = data[2]['csid_lab']
    lab_label = data[2]['y_all']
    # gestureDataLoader = gestureDataLoader( data_path = 'D:/OneShotGestureRecognition/20181116')
    # data,tData = gestureDataLoader.getTripletTrainBatcher( isOneShotTask=True, nShots=5 )
    # a = data.reshape(data.shape[0]*data.shape[1],data.shape[2],data.shape[3])
    # triplets = gestureDataLoader.getTripletTrainBatcher()
    # generator = gestureDataLoader.tripletsDataGenerator()
