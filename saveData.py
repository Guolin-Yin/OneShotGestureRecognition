from Preprocess.SignalPreprocess import *
from scipy.io import savemat, loadmat
import scipy.io as sio
import os

def preprocessData( data ):
    x = standardisers( data )
    y = remove_zero( x )
    z = get_deonised_data( y, cf_freq=100 )
    e = get_median_dnData( data=z, size=7, mode="array" )
    f, _ = pca_denoise( data=e, n_comp=7 )
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
loadAndSave()