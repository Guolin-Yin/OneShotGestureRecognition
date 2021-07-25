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
def pltResults(acc,acc_1):
    ways = np.arange( 2, 27, 1 )
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(   ways, acc,linestyle='dashed', marker='o', label='test on home environment' )
    ax.plot(   ways, acc_1,linestyle='dashed', marker='o', label='test on Lab (original) environment' )
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
model,trained_featureExtractor = defineModel( mode = 'Alexnet' )
# trained_featureExtractor.load_weights( 'D:\OneShotGestureRecognition\models\signFi_featureExtractor_weight_AlexNet_training_acc_0.95_on_276cls.h5' )
model.load_weights('./models/signFi_wholeModel_weight_AlexNet_training_acc_0.96_on_276cls.h5')
# data,filename = signDataLoder( dataDir=config.train_dir ).loadData()
# x = data[2]['csid_lab']
# x_amp = np.abs(x)
# x_phase = np.angle(x)
# x_all = np.concatenate((x_amp,x_phase),axis = 2)
# label_lab = data[2]['y_all']
# _,_,test_data,test_labels = getTrainTestSplit( data=x_all, labels=label_lab )
testSign = signDataLoder( dataDir=config.train_dir )
x_all, y_all = testSign.getFormatedData( source = 'home' )
_,_,test_data,test_labels = testSign.getTrainTestSplit( data=x_all, labels=y_all,N_train_classes =  250,
                                                        N_samples_per_class= 10)
test_labels = to_categorical(y_all - 1,num_classes=int(np.max(test_labels)))
model.evaluate(x_all,test_labels)
# testOneshot = trainTestModel()
# test_acc = testOneshot.signTest(test_data=test_data,test_labels = test_labels,
#                                 N_test_sample = 1000,embedding_model = trained_featureExtractor,
#                                 isOneShotTask = True)
# pltResults(np.asarray(test_acc),np.asarray(shot7to26))
# shot7to26 = [100.0,100.0,100.0,100.0,99.90,100.0,
#  100.0,
#  100.0,
#  99.9,
#  100.0,
#  99.8,
#  100.0,
#  100.0,
#  99.9,
#  100.0,
#  99.7,
#  99.6,
#  99.8,
#  99.6,
#  100.0,
#  99.9,
#  99.9,
#  99.9,
#  100.0,
#  100.0]
# test_on_home = [99.2,
#  98.8,
#  97.7,
#  96.39999999999999,
#  96.39999999999999,
#  95.8,
#  96.0,
#  95.0,
#  92.9,
#  93.7,
#  92.9,
#  92.4,
#  92.2,
#  90.9,
#  91.9,
#  91.8,
#  88.7,
#  89.0,
#  87.9,
#  87.0,
#  89.1,
#  87.9,
#  88.9,
#  87.6,
#  87.0]

# support_set, query_set = getOneshotTaskData( test_data, test_labels, nway=16 )
# network.summary()
# data,labels = DirectLoadData( dataDir = config.train_dir )
# X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.1)
# model.load_weights('./models/similarity_whole_model_weights_task3_single_link.h5')
# model.evaluate(reshapeData(X_train),y_train)
