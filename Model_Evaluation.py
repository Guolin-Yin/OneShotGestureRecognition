import tensorflow as tf
import numpy as np
from gestureClassification import Testing,defineModel,reshapeData,signTest,getTrainTestSplit
from Preprocess.gestureDataLoader import signDataLoder
from SiameseNetworkWithTripletLoss import SiamesWithTriplet
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from Config import getConfig
'''Initialization parameters'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
config = getConfig()
def remove_dense(model):
    encoder = Model(inputs=model.input, outputs= model.get_layer('feature_layer').output)
    return encoder
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
trained_featureExtractor.load_weights( './models/signFi_network_whole_network_structure_acc=0.91_160classes.h5')
data,filename = signDataLoder( dataDir=config.train_dir ).loadData()

x = data[2]['csid_lab']
x_amp = np.abs(x)
x_phase = np.angle(x)
x_all = np.concatenate((x_amp,x_phase),axis = 2)
label_lab = data[2]['label_lab']
_,_,test_data,test_labels = getTrainTestSplit(x_all = x_all,label_lab = label_lab)
signTest(test_data,test_labels,N_test_sample = 1000,embedding_model = trained_featureExtractor,isOneShotTask = True)

# network.summary()
# data,labels = DirectLoadData( dataDir = config.train_dir )
# X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.1)
# model.load_weights('./models/similarity_whole_model_weights_task3_single_link.h5')
# model.evaluate(reshapeData(X_train),y_train)
