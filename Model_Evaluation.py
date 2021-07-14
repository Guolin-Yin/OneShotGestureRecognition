import tensorflow as tf
from gestureClassification import Testing,defineModel,loadData,reshapeData
from SiameseNetworkWithTripletLoss import SiamesNetworkTriplet_2
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



#Define and load the trained model
model,trained_featureExtractor = defineModel( dataDir=config.train_dir )
trained_featureExtractor.load_weights( './models/similarity_model_weights.h5' )
trained_featureExtractor.summary()
Testing(test_dir = evaluation_Dir,embedding_model = trained_featureExtractor,N_test_sample = 100)
'''
Trained feature extractor performance on evaluation dataset:
Checking 2 way accuracy....
Accuracy 55.10
Checking 3 way accuracy....
Accuracy 42.70
Checking 4 way accuracy....
Accuracy 33.50
Checking 5 way accuracy....
Accuracy 30.10
Checking 6 way accuracy....
Accuracy 24.50
rained feature extractor performance on training dataset:
'''
data,labels = loadData( dataDir = config.eval_dir )
X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.1)
model.load_weights('./models/similarity_whole_model_weights.h5')
# model.evaluate(reshapeData(X_train),y_train)
