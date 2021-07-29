import tensorflow as tf
import numpy as np
from gestureClassification import *
from Preprocess.gestureDataLoader import signDataLoder
from tensorflow.keras.layers import Dense,Softmax,Input, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
config = getConfig()
testSign = signDataLoder( dataDir=config.train_dir )
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
    def __init__( self):
        self.trained_featureExtractor = _getFeatureExtractor( )
        self.trained_featureExtractor.trainable = False
        builTuningModel()
    def _cosineSim( self, x ):
        Support_embedding, Query_embedding = x
        sim = cosine_similarity( Support_embedding, Query_embedding )
        return sim
    def builTuningModel( self ):
        Support_input = Input( self.input_shape, name="Support_input" )
        Query_input = Input( self.input_shape, name="Query_input" )
        Support_embedding = self.trained_featureExtractor(Support_input)
        Query_embedding = self.trained_featureExtractor(Query_input)
        cosSim_layer = Lambda( self._cosineSim )([Support_embedding,Query_embedding])
        self.embedding_model = Model( inputs=[ Support_input, Query_input ], outputs=cosSim_layer )
    def predict( self, Support_input,Query_input ):
        sim = self.model.predict( Support_input,Query_input )
        return sim
    def original_predictions( self, Support_input,Query_input ):
        support_set_embedding = embedding_model.predict( np.asarray( Support_input ) )
        query_set_embedding = embedding_model.predict( np.expand_dims( Query_input, axis=0 ) )
        sim = cosine_similarity( support_set_embedding, query_set_embedding )
        return sim
if __name__ == '__main__':
    print('start')
    fine_tune_data,fine_tune_labels = getFineTuneData()
    fineTuningModel = fineTuningModel()
    sim = fineTuningModel.predict( Support_input,Query_input )
    sim_ori = fineTuningModel.original_predictions( Support_input,Query_input )

    # # support_set_embedding = getEmbeddingVectors(fine_tune_data)
    # labels = to_categorical(fine_tune_labels-1,num_classes=5)
    # fc = fineTuneFC(fine_tune_data,labels)
    # encoder = remove_dense(fc)
    # save_path = f'./models/fc_fineTuned.h5'
    # encoder.save_weights( save_path )

    # testFineTune()

