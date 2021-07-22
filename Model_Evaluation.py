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
'''Initialization parameters'''
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)
config = getConfig()
def remove_dense(model):
    encoder = Model(inputs=model.input, outputs= model.get_layer('feature_layer').output)
    return encoder
def pltResults(acc):
    ways = np.arange( 2, 27, 1 )
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(  acc, ways,linestyle='dashed', marker='o', label='Traninng on 250 classes' )
    ax.set_ylim( 80, 102 )
    ax.set_title( 'Accuracy change with number of new classes increase' )
    ax.set_xlabel( 'Number of new classes' )
    ax.set_ylabel( 'Accuracy' )
    ax.legend( )
#Define and load the trained model
# model,trained_featureExtractor = defineModel(  )
# trained_featureExtractor.load_weights( './models/similarity_featureExtractor_weights_task2_single_link_16class_half_samples.h5' )
# trained_featureExtractor.summary()
# Testing( test_dir=config.eval_dir,
#          embedding_model=trained_featureExtractor,
#          N_test_sample=1000,
#          isOneShotTask=True )
# Testing on signFi
_,trained_featureExtractor = defineModel( mode = '2D' )
trained_featureExtractor.load_weights( './models/signFi_featureExtractor_weight_model_training_acc_0.94_on_250cls.h5')
# data,filename = signDataLoder( dataDir=config.train_dir ).loadData()
# x = data[2]['csid_lab']
# x_amp = np.abs(x)
# x_phase = np.angle(x)
# x_all = np.concatenate((x_amp,x_phase),axis = 2)
# label_lab = data[2]['y_all']
# _,_,test_data,test_labels = getTrainTestSplit( data=x_all, labels=label_lab )
testSign = signDataLoder( dataDir=config.train_dir )
x_all, y_all = testSign.getFormatedData( )
_,_,test_data,test_labels = testSign.getTrainTestSplit( data=x_all, labels=y_all,N_train_classes =  config.N_train_classes )
testOneshot = trainTestModel()
test_acc = testOneshot.signTest(test_data=test_data,test_labels = test_labels,
                                N_test_sample = 1000,embedding_model = trained_featureExtractor,
                                isOneShotTask = True)
pltResults(test_acc)
# support_set, query_set = getOneshotTaskData( test_data, test_labels, nway=16 )
# network.summary()
# data,labels = DirectLoadData( dataDir = config.train_dir )
# X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.1)
# model.load_weights('./models/similarity_whole_model_weights_task3_single_link.h5')
# model.evaluate(reshapeData(X_train),y_train)
