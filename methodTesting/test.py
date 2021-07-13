import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, concatenate, Lambda
from Preprocess.gestureDataLoader import gestureDataLoader
import numpy as np

class SiamesNetworkTriplet:
    def __init__( self,batch_size,data_dir):
        self.batch_size = batch_size
        self.gestureDataLoader = gestureDataLoader(
                data_path=data_dir, batch_size=self.batch_size )
        # self.summary_writer = tf.summary.create_file_writer( tensorboard_log_path )
        self.input_shape = [90,1600,1]
        self.learning_rate = 0.001
        self.emb_size = 10
        self.TripletNetwork( )

    def triplet_loss( self, alpha ):
        def loss( y_true, y_pred ):
            anc, pos, neg = y_pred[ :, :emb_size ], y_pred[ :, emb_size:2 * emb_size ], y_pred[ :, 2 * emb_size: ]
            distance1 = tf.sqrt( tf.reduce_sum( tf.pow( anc - pos, 2 ), 1, keepdims=True ) )
            distance2 = tf.sqrt( tf.reduce_sum( tf.pow( anc - neg, 2 ), 1, keepdims=True ) )

            return tf.reduce_mean( tf.maximum( distance1 - distance2 + alpha, 0. ) )
        return loss
    def _buildModel(self):
        # emb_size = 128

        network = Sequential( )
        network.add( Conv2D( 64, (7, 7), activation='relu',
                             input_shape=self.input_shape,
                             kernel_initializer='he_uniform',
                              ) )
        network.add( MaxPooling2D( ) )
        network.add( Conv2D( 32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                              ) )
        network.add( MaxPooling2D( ) )
        network.add( Conv2D( 16, (3, 3), activation='relu', kernel_initializer='he_uniform',
                              ) )
        network.add( Flatten( ) )
        network.add( Dense( 16, activation='relu',

                            kernel_initializer='he_uniform' ) )

        network.add( Dense( self.emb_size, activation='sigmoid',

                            kernel_initializer='he_uniform' ) )
        network.add( Lambda( lambda x: tf.keras.backend.l2_normalize( x, axis=-1 ) ) )
        return network
    def TripletNetwork( self, margin=0.2 ):
        '''
        Define the Keras Model for training
            Input :
                input_shape : shape of input images
                ten_ges_embedding_network : Neural ten_ges_embedding_network to train outputing embeddings
                margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
            https://github.com/pranjalg2308/siamese_triplet_loss/blob/master/Siamese_With_Triplet_Loss.ipynb
            https://keras.io/examples/vision/siamese_network/
            https://github.com/CrimyTheBold/tripletloss/blob/master/02%20-%20tripletloss%20MNIST.ipynb
        '''

        network = self._buildModel( )
        # Force the encoding to live on the d-dimentional hypershpere

        # Define the tensors for the three input images
        anchor_input = Input( self.input_shape, name="anchor_input" )
        positive_input = Input( self.input_shape, name="positive_input" )
        negative_input = Input( self.input_shape, name="negative_input" )

        # Generate the encodings (feature vectors) for the three images
        encoded_a = network( anchor_input )
        encoded_p = network( positive_input )
        encoded_n = network( negative_input )

        out = concatenate( [ encoded_a, encoded_p, encoded_n ], axis=1 )
        # out = DistanceLayer()(encoded_a,encoded_p,encoded_n)

        # TripletLoss Layer
        # loss_layer = TripletLossLayer( alpha=margin, name='triplet_loss_layer' )( [ encoded_a, encoded_p, encoded_n ] )
        # Connect the inputs with the outputs
        self.model = Model( inputs=[ anchor_input, positive_input, negative_input ], outputs=out )
        # self.model.add_loss(CustomMSE())
        optimizer = tf.keras.optimizers.Adam(lr = 0.001)

        self.model.compile(loss=None,optimizer=optimizer )
        # self.model.compile( loss=self.tripletFunc( alpha=margin ), optimizer=optimizer )
        self.model.summary( )
        return self.model

    def data_generator( self ):
        while True:
            x = self.gestureDataLoader.getTripletTrainBatcher( )
            y = np.zeros( (self.batch_size, 3 * self.emb_size) )
            yield x, y
if __name__ == '__main__':
    net = SiamesNetworkTriplet( 32, data_dir ='../20181115/' )
    model = net.TripletNetwork()
    generator = net.data_generator()

    epochs = 50

    steps_per_epoch = 100

    history = model.fit(
            generator,
            epochs=epochs, steps_per_epoch=steps_per_epoch,
            verbose=True
    )