from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Lambda, ZeroPadding2D,Conv2D,\
    MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from Config import getConfig
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
config = getConfig()
def learning_rate_schedule(process,init_learning_rate = 0.01,alpha = 10.0 , beta = 0.75):
    num = (process//50)

    return init_learning_rate*(0.1)**num
    # return init_learning_rate /(1.0 + alpha * process)**beta
def grl_lambda_schedule(process,gamma=10.0):

    return 2.0 / (1.0+np.exp(-gamma*process)) - 1.0
@tf.custom_gradient
def gradient_reversal(x,alpha=1.0):
	def grad(dy):
		return -dy * alpha, None
	return x, grad
class GradientReversalLayer(Layer):

	def __init__(self,**kwargs):
		super(GradientReversalLayer,self).__init__(kwargs)

	def call(self, x,alpha=1.0):
		return gradient_reversal(x,alpha)
class AdversarialNetwork():
    def __init__(self,config):
        self.config = config
        self.modelObj = models()
    def getFeatureExtractor(self,model):
        feature_extractor = Model(inputs=model.input, outputs= model.get_layer('lambda_layer').output)
        return feature_extractor
    def buildAdvModel( self, ):
        input = Input( self.config.input_shape, name = 'data input' )
        feature_extractor = self.modelObj.buildFeatureExtractor( mode = 'adv', input = input )
        # feature_extractor = self.feature_extractor( input )
        '''gesture_classifier'''
        # f1 = Dense( units = 1024, name = 'g_1' )( feature_extractor )
        # f2 = Dense( units = 512, name = 'g_2' )( f1 )
        # f3 = Dense( units = 256, name = 'g_3' )( f2 )
        gesture_classifier = Dense(
                units = self.config.N_base_classes + self.config.N_novel_classes,
                bias_regularizer = regularizers.l2( 4e-4 ), name = 'gesture_classifier'
                )( feature_extractor )
        gesture_classifier_out = Softmax( name = 'gesture_classifier_out' )( gesture_classifier )
        '''domain_discriminator'''
        g = GradientReversalLayer()(feature_extractor)
        f = Dense(units = 1024, name = 'D_1')(g)
        e = Dense( units = 512, name = 'D_2' )( f )
        m = Dense( units = 256, name = 'D_3' )( e )
        domain_discriminator = Dense(
                units = 2,
                bias_regularizer = regularizers.l2( 4e-4 ), name = 'domain_discriminator'
                )( m )
        domain_discriminator_out = Softmax( name = 'domain_discriminator_out' )( domain_discriminator )
        adv_model = Model( inputs = input, outputs = [ gesture_classifier_out ] )
        adv_model.summary()
        return adv_model
    def buildFeatureExtractor( self,):
        model = tf.keras.Sequential(
                [
                        # First conv layer
                        Conv2D(
                                filters = 96, kernel_size = (11, 5), strides = 2, input_shape = config.input_shape,
                                padding = 'valid',
                                activation = 'relu', name = 'Conv_1'
                                ),

                        MaxPooling2D( pool_size = 3, strides = 1, name = 'Maxpool_1' ),
                        ZeroPadding2D( padding = 2, name = 'Padding_layer_1' ),
                        Conv2D( filters = 256, kernel_size = 5, strides = 1, padding = 'valid', name = 'Conv_2' ),
                        MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_2' ),
                        ZeroPadding2D( padding = 1, name = 'Padding_leayer_2' ),
                        Conv2D(
                                filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                                name = 'Conv_3'
                                ),
                        ZeroPadding2D( padding = 1, name = 'Padding_layer_3' ),
                        Conv2D(
                                filters = 384, activation = 'relu', kernel_size = 3, strides = 1, padding = 'valid',
                                name = 'Conv_4'
                                ),
                        ZeroPadding2D( padding = 1, name = 'Padding_layer_4' ),
                        Conv2D(
                                filters = 256, activation = 'relu', kernel_size = (4, 3), strides = 1,
                                padding = 'valid',
                                name = 'Conv_5'
                                ),
                        MaxPooling2D( pool_size = 3, strides = 2, name = 'Maxpool_3' ),
                        Dropout( 0.5 ),
                        Flatten( ),
                        Dense( units = 4096, name = 'FC_1' ),
                        Dense( units = 4096, name = 'FC_2' ),
                        Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )
                        ]
                )
        return model
    def buildSignClassifier( self):
        model = tf.keras.Sequential(
                [
                        Dense( 1024, activation = 'relu' ),
                        Dense( self.config.N_base_classes, activation = 'softmax', name = "sign_Pred" ),
                        ]
                )
        return model
    def buildDomainClassifier( self):
        model = tf.keras.Sequential(
                [
                        # GradientReversalLayer(),
                        tf.keras.layers.Dense( 100 ),
                        # tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation( 'relu' ),
                        # tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense( 6, activation = 'softmax', name = "domain_cls_pred" )
                        ]
                )
        return model
class models:
    def __init__( self ):
        pass
    def buildFeatureExtractor( self ,mode:str ='Alexnet',input = None):
        if mode =='Alexnet':
            input = Input( config.input_shape, name = 'input_layer' )
            conv_1 = Conv2D(
                    filters = 96, kernel_size = (11, 5), strides = 2, input_shape = config.input_shape,
                    padding = 'valid',
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
            FC_1 = Dense( units = 256, name = 'FC_1' )( ft )
            FC_2 = Dense( units = 1280, name = 'FC_2' )( FC_1 )
            output = Lambda( lambda x: K.l2_normalize( x, axis = -1 ),name = 'lambda_layer' )( FC_2 )
            feature_extractor = Model( inputs = input, outputs = output )
        elif mode =='adv':
            conv_1 = Conv2D(
                    filters = 96, kernel_size = (11, 5), strides = 2, input_shape = config.input_shape,
                    padding = 'valid',
                    activation = 'relu', name = 'Conv_1'
                    )(input)
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
            FC_2 = Dense( units = 3000, name = 'FC_2' )( FC_1 )
            feature_extractor = Lambda( lambda x: K.l2_normalize( x, axis = -1 ), name = 'lambda_layer' )( FC_2 )
        return feature_extractor
    def buildTuneModel( self ,config, isTest:bool = False,pretrained_feature_extractor = None):
        if isTest:
            feature_extractor = self.buildFeatureExtractor( mode = 'Alexnet')
            fc = Dense( units = config.N_novel_classes, name = "fine_tune_layer" )( feature_extractor.output )
            output = Softmax( )( fc )
            fine_Tune_model = Model( inputs = feature_extractor.input, outputs = output )
        if not isTest:
            try:
                fc = Dense( units = config.N_novel_classes,
                        bias_regularizer = regularizers.l2( 1e-4 ),
                        name = "fine_tune_layer" )(pretrained_feature_extractor.output )
                output = Softmax( )( fc )
                fine_Tune_model = Model( inputs = pretrained_feature_extractor.input, outputs = output )
            except AttributeError:
                print("The feature extractor has not been passed!!!!!!!!!!!!!!")
        fine_Tune_model.summary( )
        return fine_Tune_model
    def lr_scheduler(self, epoch, lr):
        if epoch < 100:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
if __name__ == "__main__":
    m = models()
    net = m.buildFeatureExtractor()
    net.summary()