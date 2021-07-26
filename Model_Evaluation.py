import tensorflow as tf
import numpy as np
from gestureClassification import *
from Preprocess.gestureDataLoader import signDataLoder
from SiameseNetworkWithTripletLoss import SiamesWithTriplet
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from Config import getConfig
from matplotlib.ticker import MaxNLocator
# from methodTesting.plotResults import pltResults

config = getConfig()
def remove_dense(model):
    encoder = Model(inputs=model.input, outputs= model.get_layer('feature_layer').output)
    return encoder
def pltResults(acc):
    ways = np.arange( 2, 27, 1 )
    ways2 = np.arange( 2, 26, 1 )
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot( ways, acc[ 0 ],linestyle='dashed', marker='o', label='test on lab environment same user (New sign)' )
    ax.plot( ways, acc[ 1 ], linestyle='dashed', marker='o', label='test on lab environment same user' )
    ax.plot( ways, acc[ 2 ], linestyle='dashed', marker='o', label='test on different environment same user (New sign)' )
    ax.plot( ways, acc[ 3 ], linestyle='dashed', marker='o', label='test on different environment different user ' )
    ax.plot( ways2, acc[ 4 ], linestyle='dashed', marker='o', label='test on different environment different user (New sign)' )
    ax.set_ylim( 0, 102 )
    ax.set_title( 'Feature extractor trained on lab environment with 250 classes' )
    ax.set_xlabel( 'Number of new classes' )
    ax.set_ylabel( 'Accuracy' )
    ax.legend( )
def record():
    test_on_lab = [ 100.0, 100.0, 100.0, 100.0, 99.90, 100.0,100.0,100.0,99.9,100.0,99.8,100.0,100.0,99.9,100.0,99.7,99.6,99.8,99.6,100.0,99.9,99.9,99.9,100.0,100.0 ]
    test_on_home = [ 99.2,
                     98.8,
                     97.7,
                     96.39999999999999,
                     96.39999999999999,
                     95.8,
                     96.0,
                     95.0,
                     92.9,
                     93.7,
                     92.9,
                     92.4,
                     92.2,
                     90.9,
                     91.9,
                     91.8,
                     88.7,
                     89.0,
                     87.9,
                     87.0,
                     89.1,
                     87.9,
                     88.9,
                     87.6,
                     87.0 ]
    test_on_user1_trained_sign = [ 84.6,
                                   80.30000000000001,
                                   82.69999999999999,
                                   81.6,
                                   79.5,
                                   79.9,
                                   80.30000000000001,
                                   80.4,
                                   78.4,
                                   82.69999999999999,
                                   80.30000000000001,
                                   80.10000000000001,
                                   78.0,
                                   79.4,
                                   80.4,
                                   78.5,
                                   77.5,
                                   79.80000000000001,
                                   77.8,
                                   77.3,
                                   78.7,
                                   79.10000000000001,
                                   78.60000000000001,
                                   78.8,
                                   80.0 ]
    test_on_user1_unseen_sign = [ 64.9, 47, 37.9, 34, 29.5, 27.4, 25.1, 22.5, 19.1, 18.4, 17.1, 15.3, 14.3, 14.7, 13.8,
                                  14.5, 15.2, 11.5,
                                  12.4, 10.3, 8.5, 10, 11.6, 7.7 ]
    test_on_user5_trained_sign = [ 99.3,
                                   99.5,
                                   99.2,
                                   98.2,
                                   98.7,
                                   97.89999999999999,
                                   98.0,
                                   98.5,
                                   98.0,
                                   97.89999999999999,
                                   97.2,
                                   97.1,
                                   97.0,
                                   97.1,
                                   97.5,
                                   97.2,
                                   95.6,
                                   96.39999999999999,
                                   94.69999999999999,
                                   96.8,
                                   96.7,
                                   97.2,
                                   95.3,
                                   96.0,
                                   96.89999999999999 ]

    pltResults( [ test_on_lab, test_on_user5_trained_sign, test_on_home, test_on_user1_trained_sign,
                  test_on_user1_unseen_sign ] )
def split(x_all,y_all):
    start = np.where( y_all == 254 )[ 0 ]
    # end = start + 25
    test_labels_user1 = np.zeros( (250, 1), dtype=int )
    test_data_user1 = np.zeros((250,200,60,3))
    count = 0
    for i in start:
        test_labels_user1[ count:count + 25 ] = y_all[ i:i + 25 ]
        test_data_user1[count:count + 25] = x_all[i:i + 25]
        count += 25
    return [test_data_user1,test_labels_user1]
def OneShotPerformanceTest():
    _,trained_featureExtractor = defineModel( mode = 'Alexnet' )
    trained_featureExtractor.load_weights( 'D:\OneShotGestureRecognition\models\signFi_featureExtractor_weight_AlexNet_training_acc_0.95_on_250cls.h5' )
    testSign = signDataLoder( dataDir=config.train_dir )
    x_all, y_all = testSign.getFormatedData( source='home' )
    # test_data, test_labels = split( x_all, y_all )
    test_data, test_labels, _, _ = testSign.getTrainTestSplit( data=x_all, labels=y_all,
                                                               N_train_classes=26,
                                                               N_samples_per_class= 10)
    testOneshot = trainTestModel( )
    test_acc = testOneshot.signTest( test_data=test_data, test_labels=test_labels,
                                     N_test_sample=1000, embedding_model=trained_featureExtractor,
                                     isOneShotTask=True )
    return test_acc
def CnnModelTesting():
    model, _ = defineModel( mode='Alexnet' )
    model.load_weights('./models/signFi_wholeModel_weight_AlexNet_training_acc_0.96_on_276cls.h5')
    testSign = signDataLoder( dataDir=config.train_dir )
    x_all, y_all = testSign.getFormatedData( source='lab_other' )
    test_labels_1 = to_categorical(y_all[0:1250] - 1,num_classes=int(np.max(y_all)-2))
    test_labels_2 = to_categorical( y_all[ 1250:1500 ] - 3, num_classes=int( np.max( y_all ) - 2 ) )
    test_labels = np.concatenate((test_labels_1,test_labels_2),axis = 0)
    model.evaluate(x_all,test_labels)
if __name__ == '__main__':
    OneShotPerformanceTest()
    # CnnModelTesting()

