from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, Lambda,Dot,concatenate,ZeroPadding2D,Conv2D,\
    MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from Config import getConfig
config = getConfig()
class models:
    def __init__( self ):
        pass

    def buildFeatureExtractor( self ,mode:str ='Alexnet'):
        if mode =='Alexnet':
            input = Input( config.input_shape, name = 'input_layer' )
            conv_1 = Conv2D(
                    filters = 96, kernel_size = (11, 5), strides = 2, input_shape = config.input_shape, padding = 'valid',
                    activation = 'relu', name = 'Conv_1'
                    )( input )
            MP_1 = MaxPooling2D( pool_size = 3, strides = 1, name = 'Maxpool_1' )( conv_1 )

            PD_1 = ZeroPadding2D( padding = 2, name = 'Padding_layer_1' )( MP_1 )
            conv_2 = Conv2D( filters = 256, kernel_size = 5, strides = 1, padding = 'valid', name = 'Conv_2' )( PD_1 )
            MP_2 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_2' )( conv_2 )
            Padding_leayer_2 = ZeroPadding2D( padding = 1, name = 'Padding_leayer_2' )( MP_2 )
            Conv_3 = Conv2D(
                    filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                    name = 'Conv_3'
                    )( Padding_leayer_2 )
            Padding_layer_3 = ZeroPadding2D( padding = 1, name = 'Padding_layer_3' )( Conv_3 )
            Conv_4 = Conv2D(
                    filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                    name = 'Conv_4'
                    )( Padding_layer_3 )
            Padding_layer_4 = ZeroPadding2D( padding = 1, name = 'Padding_layer_4' )( Conv_4 )
            Conv_5 = Conv2D(
                    filters = 256, activation = 'relu', kernel_size = (4, 3), strides = 1, padding = 'valid',
                    name = 'Conv_5'
                    )( Padding_layer_4 )
            Maxpool_3 = MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_3' )( Conv_5 )
            dp = Dropout( 0.5 )( Maxpool_3 )
            ft = Flatten( )( dp )
            FC_1 = Dense( units = 4096, name = 'FC_1' )( ft )
            FC_2 = Dense( units = 4096, name = 'FC_2' )( FC_1 )
            output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ) )( FC_2 )
            feature_extractor = Model( inputs = input, outputs = output )
        return feature_extractor
	# def builPretrainModel( self, mode:str ='Alexnet' ):
	# 	if mode == 'Alexnet':
	# 		input = Input( config.input_shape, name = 'data input' )
	# 		encoded_model = network( input )
	# 		full_connect = Dense( units = config.N_train_classes )( encoded_model )
	# 		output = Softmax( )( full_connect )
	# 		model = Model( inputs = input, outputs = output )
	# 		# Complie model
	# 		optimizer = tf.keras.optimizers.SGD(
	# 				learning_rate = config.lr, momentum = 0.9
	# 				)
	# 		model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
	# 		# model.summary( )
	# 		return model, network
    def buildTuneModel( self ):
        feature_extractor = self.buildFeatureExtractor( mode = 'Alexnet')
        fc = Dense( units = 25, name = "fine_tune_layer" )( feature_extractor.output )
        output = Softmax( )( fc )
        fine_Tune_model = Model( inputs = feature_extractor.input, outputs = output )
        fine_Tune_model.summary( )
        return fine_Tune_model

