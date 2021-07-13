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
    #                     labels.append(to_categorical(gestureType, num_classes = 9))
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
        sim_gesture_6ges = ['20181115','20181109','Pre_16']
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
    def _mapPathToDataAndLabels(self,path:list, is_one_shot_task:bool,is_triplet_loss:bool):
        if not is_triplet_loss:
            num_of_pairs = int(len(path)/2)
            pairs_of_samples = [ np.zeros( (num_of_pairs, self.num_subcarriers, self.len_signals) ) for i in range( 2 ) ]
            # pairs_of_samples = [ np.zeros( num_of_pairs, 90, 1600 ) for i in range( 2 ) ]
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
            # buff = [ ]
            # for in_trip in range(len(triplets)):
            #     buf = [ ]
            #     for j in range(triplets[0].shape[0]):
            #         buf.append(self._preprocessData(triplets[in_trip][j,:,:]))
            #     buff.append(np.asarray(buf))
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
        data, labels = self._mapPathToDataAndLabels(batch_gesture_path,is_one_shot_task = False,is_triplet_loss=False)
        return data,labels
    def getTripletTrainBatcher(self):
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
        data = self._mapPathToDataAndLabels( triplets_path, is_one_shot_task=False, is_triplet_loss=True )
        return data
    def tripletsDataGenerator(self):
        while True:

            a,p,n = self.getTripletTrainBatcher()
            labels = np.ones(self.batch_size)
            yield [a,p,n], labels

if __name__ == '__main__':
    gestureDataLoader = gestureDataLoader( data_path = 'D:/OneShotGestureRecognition/20181116')
    data = gestureDataLoader.getTripletTrainBatcher()
    # triplets = gestureDataLoader.getTripletTrainBatcher()
    # generator = gestureDataLoader.tripletsDataGenerator()
    # x = standardisers( data )
    # y = remove_zero( x )
    # z = get_deonised_data( y, cf_freq=100 )
    # e = get_median_dnData( data=z, size=7, mode="array" )
    # f,_ = pca_denoise(data = e, n_comp = 7)