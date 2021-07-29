import tensorflow as tf
import numpy as np
from gestureClassification import *
from Preprocess.gestureDataLoader import signDataLoder
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, Lambda,Dot,concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig

# x_all, y_all = testSign.getFormatedData( source='lab_other' )
def _getFeatureExtractor():
    _, trained_featureExtractor = defineModel( mode='Alexnet' )
    trained_featureExtractor.load_weights('./models/signFi_wholeModel_weight_AlexNet_training_acc_0.90_on_125cls_user1to4.h5' )
    return trained_featureExtractor
def getEmbeddingVectors(input):
    trained_featureExtractor = _getFeatureExtractor( )
    support_set_embedding = trained_featureExtractor.predict(input)
    return support_set_embedding
def buildFc():
    trained_featureExtractor = _getFeatureExtractor()
    trained_featureExtractor.trainable = False

    input = Input( config.input_shape, name='data input' )
    featureExtractor = trained_featureExtractor( input )
    fc_layer = Dense( units=4096, name='fc_feature' )( featureExtractor )
    fc_layer_out = Dense(units = 5,name = 'fc_out')(fc_layer)
    output = Softmax()(fc_layer_out)

    fc = Model(inputs = input,outputs = output)
    return fc
def fineTuneFC(support_set_embedding,labels):
    fc = buildFc()
    optimizer = tf.keras.optimizers.SGD( learning_rate=config.lr, momentum=0.9 )
    fc.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
    fc.fit(support_set_embedding,labels,epochs = 100,shuffle=True)
    return fc
def getFineTuneData():
    x_all, y_all = testSign.getFormatedData( source='lab_other' )
    x_all = x_all[0:1250]
    y_all = y_all[0:1250]

    start,_ = np.where(y_all == 1)
    fine_tune_data = np.zeros((25,200,60,3))
    fine_tune_labels = np.zeros( (25, 1),dtype = int )
    count = 0
    for i in start[0:5]:
        fine_tune_labels[count:count+5,:] = y_all[i:i+5,:]
        fine_tune_data[count:count+5,:,:,:] = x_all[i:i+5,:,:,:]
        count += 5
    return fine_tune_data,fine_tune_labels
def remove_dense(model):
    encoder = Model(inputs=model.input, outputs= model.get_layer('fc_feature').output)
    return encoder
def testFineTune():
    testOneshot = trainTestModel( )
    testSign = signDataLoder( dataDir=config.train_dir )
    encoder = buildFc( )
    encoder = remove_dense(encoder)
    encoder.load_weights('./models/fc_fineTuned.h5')
    encoder.summary()
    x_all, y_all = testSign.getFormatedData( source='lab_other' )
    test_data = x_all[ 1250:1500 ]
    test_labels = y_all[ 1250:1500 ]
    test_acc = testOneshot.signTest( test_data=test_data, test_labels=test_labels,
                                     N_test_sample=1000, embedding_model=encoder,
                                     isOneShotTask=True, mode='fix' )
    return test_acc
class fineTuningModel:
    def __init__( self ):
        self.trained_featureExtractor = self._getFeatureExtractor( )
        self.trained_featureExtractor.trainable = False
        self.input_shape = config.input_shape
        self.builTestModel( )
    def _getFeatureExtractor( self ):
        _, trained_featureExtractor = defineModel( mode='Alexnet' )
        trained_featureExtractor.load_weights(
            './models/signFi_wholeModel_weight_AlexNet_training_acc_0.90_on_125cls_user1to4.h5' )
        return trained_featureExtractor
    def builTestModel( self ):
        Support_input = Input( self.input_shape, name="Support_input" )
        Query_input = Input( self.input_shape, name="Query_input" )
        Support_embedding = self.trained_featureExtractor( Support_input )
        Query_embedding = self.trained_featureExtractor(Query_input)
        cosSim_layer = Dot( axes=1, normalize=True )([Support_embedding,Query_embedding])
        self.embedding_model = Model( inputs=[ Support_input, Query_input  ], outputs=cosSim_layer )
    def predict( self, Support_input=None, Query_input = None):
        sim = self.embedding_model.predict( [Support_input,Query_input] )
        return sim
    def buildFineTuningModel( self ):



if __name__ == '__main__':
    print('start')
    config = getConfig( )
    fineTuning = fineTuningModel( )
    testSign = signDataLoder( dataDir=config.train_dir )

    fine_tune_data, fine_tune_labels = getFineTuneData( )
    Support_input = np.expand_dims(fine_tune_data[1:2],axis=0)
    Query_input = np.expand_dims(fine_tune_data[6:7],axis=0)
    sim = fineTuning.predict( Support_input = fine_tune_data[0:2], Query_input = fine_tune_data[5] )


    # # support_set_embedding = getEmbeddingVectors(fine_tune_data)
    # labels = to_categorical(fine_tune_labels-1,num_classes=5)
    # fc = fineTuneFC(fine_tune_data,labels)
    # encoder = remove_dense(fc)
    # save_path = f'./models/fc_fineTuned.h5'
    # encoder.save_weights( save_path )

    # testFineTune()

# input_a = Input(shape=(input_dim, 1))
# input_b = Input(shape=(input_dim, 1))
#
# cos_distance = merge([input_a, input_b], mode='cos', dot_axes=1) # magic dot_axes works here!
# cos_distance = Reshape((1,))(cos_distance)
# cos_similarity = Lambda(lambda x: 1-x)(cos_distance)
#
# model = Model([input_a, input_b], [cos_similarity])





