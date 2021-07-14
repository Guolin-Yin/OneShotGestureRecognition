import tensorflow as tf
# from keras.layers import Layer,Conv1D, Conv2D, Flatten,\
#     Dense,Dropout, Input, Lambda,MaxPooling2D,\
#     concatenate,BatchNormalization,MaxPooling1D
from gestureClassification import Testing
from SiameseNetworkWithTripletLoss import SiamesNetworkTriplet_2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
evaluation_Dir = 'D:/OneShotGestureRecognition/20181116'
#Define and load the trained model
# trained_featureExtractor = tf.saved_model.load( './similarity_model.h5' )
embedding = SiamesNetworkTriplet_2(batch_size=32,data_dir=evaluation_Dir,lr = 0.001)
trained_featureExtractor = embedding.build_embedding_network()
trained_featureExtractor.load_weights( 'models/similarity_model_weights.h5' )
trained_featureExtractor.summary()
Testing(test_dir = 'D:/OneShotGestureRecognition/20181115/',embedding_model = trained_featureExtractor)

embedding = SiamesNetworkTriplet_2(batch_size=32,data_dir=evaluation_Dir,lr = 0.001)
trained_featureExtractor = embedding.build_embedding_network()
trained_featureExtractor.load_weights( 'models/similarity_model_weights.h5' )
trained_featureExtractor.summary()