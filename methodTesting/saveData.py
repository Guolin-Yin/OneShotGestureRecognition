from Preprocess.SignalPreprocess import *
from scipy.io import savemat, loadmat
import scipy.io as sio
import os
import re

def preprocessData( data,reduction:bool = True ):
    x = standardisers( data )
    y = remove_zero( x )
    z = get_deonised_data( y, cf_freq=100 )
    e = get_median_dnData( data=z, size=7, mode="array" )
    if reduction == True:
        f, _ = pca_denoise( data=e, n_comp=7 )
    else:
        f = e
    return f
def loadAndSave( ):
    path = 'D:/OneShotGestureRecognition/20181116'
    filename = os.listdir( path )
    for name in filename:
        print(f'saving{name}')
        filePath = os.path.join( path, name )
        data = sio.loadmat(filePath)['csiAmplitude']
        pro_data = preprocessData(data)
        buf = {'csiAmplitude':pro_data}
        file = name+'.mat'
        savemat(os.path.join('D:\OneShotGestureRecognition\Pre_16',file), buf )
if __name__ == '__main__':
    Dir = ['D:\\Widar_dataset_matfiles\\20181116','D:\\Widar_dataset_matfiles\\20181115']
    for path in Dir:
        gesture = [ ]
        location = []
        orientation = []
        repetition = []
        link = []
        if path == 'D:\\Widar_dataset_matfiles\\20181116':
            Save_Dir = 'D:\OneShotGestureRecognition\Combined_link_dataset/20181116'
        if path == 'D:\\Widar_dataset_matfiles\\20181115':
            Save_Dir = 'D:\OneShotGestureRecognition\Combined_link_dataset/20181115'
        filename = os.listdir( path )
        for name in filename:
            gesture.append( int( re.findall( r'\d+\b', name )[ 1 ] ) )
            location.append( int( re.findall( r'\d+\b', name )[ 2 ] ) )
            orientation.append( int( re.findall( r'\d+\b', name )[ 3 ] ) )
            repetition.append( int( re.findall( r'\d+\b', name )[ 4 ] ) )
            link.append( int( re.findall( r'\d+\b', name )[ 5 ] ) )
        num_gesture_types = np.max( gesture ) - np.min( gesture ) + 1

        range_gesture = np.arange(np.min( gesture ),np.max( gesture ) + 1)
        print(f'the range of gesture is from {np.min( gesture )} to {np.max( gesture )}')
        range_location = np.arange(np.min( location ),np.max( location )+ 1)
        print(f'the range of location is from {np.min( location )} to {np.max( location )}')
        range_orientation = np.arange(np.min( orientation ),np.max( orientation )+ 1)
        print(f'the range of orientation is from {np.min( orientation )} to {np.max( orientation )}')
        range_repetition = np.arange(np.min( repetition ),np.max( repetition )+ 1)
        print(f'the range of repetition is from {np.min( repetition )} to {np.max( repetition )}')
        range_link = np.arange(np.min( link ),np.max( link )+ 1)
        print(f'the range of link is from {np.min( link )} to {np.max( link )}')
        fName = []
        for g in range_gesture:
            for loc in range_location:
                for ori in range_orientation:
                    for rp in range_repetition:
                        saveName = []
                        data = []
                        for lin in range_link:
                            # saveName.append( f'user1-{g}-{loc}-{ori}-{rp}-r{lin}.mat')
                            outName = f'user1-{g}-{loc}-{ori}-{rp}'
                            pathName = outName + f'-r{lin}' +'.mat'
                            sPath = os.path.join(path,pathName)
                            data.append( sio.loadmat( sPath )[ 'csiAmplitude' ] )
                        outData = np.concatenate( (data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ]), axis=0 )
                        buf = { 'csiAmplitude':  preprocessData(outData)  }
                        # outName = f'user1-{ g }-{loc}-{ori}-{rp}.mat'
                        print(f'saving data {outName}')
                        outFilename = outName + '.mat'
                        savemat( os.path.join( Save_Dir, outFilename ), buf )
                        fName.append(outFilename)









# fName = []
# for g in range_gesture:
#     for loc in range_location:
#         for ori in range_orientation:
#             for rp in range_repetition:
#                 saveName = []
#                 for lin in range_link:
#                     saveName.append( f'user1-{g}-{loc}-{ori}-{rp}-r{lin}.mat')
#                 fName.append(saveName)
#
# for saveIdx in range(len(fName)):
#     files = fName[saveIdx]
#     gestures = int(re.findall( r'\d+\b', files[0] )[ 1 ])
#     locations = int(re.findall( r'\d+\b', files[0] )[ 2 ])
#     orientations = int(re.findall( r'\d+\b', files[0] )[ 3 ])
#     repetitions = int(re.findall( r'\d+\b', files[0] )[ 4 ])
#     data = [ ]
#     for w in range(len(files)):
#         sPath = os.path.join( path, files[w] )
#         data.append(sio.loadmat( sPath )[ 'csiAmplitude' ])
#     outData = np.concatenate((data[0],data[1],data[2],data[3],data[4],data[5]),axis = 0)
#     buf = { 'csiAmplitude': preprocessData(outData) }
#     outName = f'user1-{ gestures }-{locations}-{orientations}-{repetitions}.mat'
#     print(f'saving data {outName}')
#     savemat( os.path.join( 'D:\OneShotGestureRecognition\Combined_link_dataset/20181115', outName ), buf )
'''
0: user id
1: gesture type, from 1 to 6
2: location, from 6 to 8
3: orientation, from 1 to 5
4: repetition, 1 to 20
5: link ID, 1 to 6
'''
