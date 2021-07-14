import tensorflow as tf
# from keras.layers import Layer,Conv1D, Conv2D, Flatten,\
#     Dense,Dropout, Input, Lambda,MaxPooling2D,\
#     concatenate,BatchNormalization,MaxPooling1D
from gestureClassification import Testing
from SiameseNetworkWithTripletLoss import SiamesNetworkTriplet_2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


evaluation_Dir = 'D:/OneShotGestureRecognition/20181115'
#Define and load the trained model
# trained_featureExtractor = tf.saved_model.load( './similarity_model.h5' )
embedding = SiamesNetworkTriplet_2(batch_size=32,data_dir=evaluation_Dir,lr = 0.001)
trained_featureExtractor = embedding.build_embedding_network()
trained_featureExtractor.load_weights( 'models/similarity_model_weights.h5' )
trained_featureExtractor.summary()
Testing(test_dir = evaluation_Dir,embedding_model = trained_featureExtractor)
'''
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
'''
new_embedding = SiamesNetworkTriplet_2(batch_size=32,data_dir=evaluation_Dir,lr = 0.001)
new_extractor = new_embedding.build_embedding_network()
new_extractor
new_extractor.summary()
Testing(test_dir = evaluation_Dir,embedding_model = new_extractor)