import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv1D, Conv2D, Flatten, Dense,Dropout, Input, Lambda,MaxPooling2D,AveragePooling2D,\
                                    concatenate,BatchNormalization,MaxPooling1D,ReLU,Softmax
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import random
from Preprocess.gestureDataLoader import gestureDataLoader
import numpy as np
from Config import getConfig
import os
# class TripletLossLayer( Layer ):
#     def __init__( self, alpha, **kwargs ):
#         self.alpha = alpha
#         super( TripletLossLayer, self ).__init__( **kwargs )
#
#     def triplet_loss( self, inputs ):
#         anchor, positive, negative = inputs
#         p_dist = K.sum( K.square( anchor - positive ), axis=-1 )
#         n_dist = K.sum( K.square( anchor - negative ), axis=-1 )
#         loss = K.sum( K.maximum( p_dist - n_dist + self.alpha, 0 ), axis=0 )
#         return loss
#
#     def call( self, inputs ):
#         loss = self.triplet_loss( inputs )
#         self.add_loss( loss )
#         print( 'Adding loss')
#         return loss
# class SiamesNetworkTriplet:
#     def __init__( self,batch_size,data_dir):
#         self.batch_size = batch_size
#         self.gestureDataLoader = gestureDataLoader(
#                 data_path=data_dir, batch_size=self.batch_size )
#         # self.summary_writer = tf.summary.create_file_writer( tensorboard_log_path )
#         self.input_shape = [7,1600]
#         self.learning_rate = 0.001
#         self.TripletNetwork( )
#     def train_siamese_network(self,number_of_iterations = 1000,evaluate_each = 1000,model_name:str =  'testing'):
#         print( "Num GPUs Available: ", len( tf.config.experimental.list_physical_devices( 'GPU' ) ) )
#         train_losses = np.zeros( shape=(evaluate_each) )
#         train_accuracies = np.zeros( shape=(evaluate_each) )
#         # training params
#         trainCount = 0
#         best_validation_accuracy = 0.0
#         best_accuracy_iteration = 0
#
#         for iter in range(number_of_iterations):
#             data = self.gestureDataLoader.getTripletTrainBatcher( )
#             train_loss = self.model.train_on_batch(data)
#             # Update learning rate and momentum
#             if (iter + 1)% 500 == 0:
#                 K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr)*0.99)
#             # if K.get_value(self.model.optimizer.momentum) < final_momentum:
#             #     K.set_value( self.model.optimizer.momentum,
#             #                  K.get_value( self.model.optimizer.momentum ) + momentum_slope )
#
#             train_losses[trainCount] = train_loss
#             # train_accuracies[trainCount] = train_acc
#             trainCount += 1
#             print( 'Iteration %d/%d: Train loss: %f, lr = %f' %
#                    (iter + 1, number_of_iterations, train_loss, K.get_value(
#                            self.model.optimizer.lr )) )
#             # perform a one shot task evaluation on the validation data
#             # if (iter + 1) % evaluate_each == 0:
#             #     num_of_run_per_gesture = 40
#             #     # validation_accuracy = self.gestureDataLoader.oneShotTest(self.model,
#             #     #                                                          support_set_size,
#             #     #                                                          number_of_run_per_gesture,
#             #     #                                                          is_validation=True)
#             #     # self._write_logs_to_tensorboard(
#             #     #         iteration, train_losses, train_accuracies,
#             #     #         validation_accuracy, evaluate_each )
#             #     trainCount = 0
#             #     if (validation_accuracy == 1.0 and train_accuracy == 0.5):
#             #         print('Early Stopping: Gradient Explosion')
#             #         print('Validation Accuracy = ' +
#             #               str(best_validation_accuracy))
#             #         return 0
#             #     elif train_accuracy == 0.0:
#             #         return 0
#             #     else:
#             #         if validation_accuracy > best_validation_accuracy:
#             #             best_validation_accuracy = validation_accuracy
#             #             best_accuracy_iteration = iter
#             #
#             #             model_json = self.model.to_json()
#             #
#             #             if not os.path.exists("./models"):
#             #                 os.makedirs("./models")
#             #             with open("models/"+model_name+ ".json","w") as json_file:
#             #                 json_file.write(model_json)
#             #             self.model.save_weights("models/"+model_name+".h5")
#             #         if iter - best_accuracy_iteration > 1000:
#             #             print(
#             #                     'Early Stopping: validation accuracy did not increase for 10000 iterations' )
#             #             print( 'Best Validation Accuracy = ' +
#             #                    str( best_validation_accuracy ) )
#             #             print( 'Validation Accuracy = ' + str( best_validation_accuracy ) )
#             #             break
#         print("The end of training process")
#         return best_validation_accuracy
#     def _buildModel(self):
#         # emb_size = 128
#         emb_size = 28
#         network = Sequential( )
#         network.add( Conv2D( 32, (7, 7), activation='relu',
#                              input_shape=self.input_shape,
#                              kernel_initializer='he_uniform',
#                              kernel_regularizer=l2( 2e-4 ) ) )
#         network.add( MaxPooling2D( ) )
#         network.add( Conv2D( 64, (3, 3), activation='relu', kernel_initializer='he_uniform',
#                              kernel_regularizer=l2( 2e-4 ) ) )
#         network.add( MaxPooling2D( ) )
#         network.add( Conv2D( 32, (3, 3), activation='relu', kernel_initializer='he_uniform',
#                              kernel_regularizer=l2( 2e-4 ) ) )
#         network.add( Flatten( ) )
#         network.add( Dense( 32, activation='relu',
#                             kernel_regularizer=l2( 1e-3 ),
#                             kernel_initializer='he_uniform' ) )
#
#         network.add( Dense( emb_size, activation='sigmoid',
#                             kernel_regularizer=l2( 1e-3 ),
#                             kernel_initializer='he_uniform' ) )
#         return network
#     def TripletNetwork( self, margin=0.2 ):
#         '''
#         Define the Keras Model for training
#             Input :
#                 input_shape : shape of input images
#                 ten_ges_embedding_network : Neural ten_ges_embedding_network to train outputing embeddings
#                 margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
#             https://github.com/pranjalg2308/siamese_triplet_loss/blob/master/Siamese_With_Triplet_Loss.ipynb
#             https://keras.io/examples/vision/siamese_network/
#             https://github.com/CrimyTheBold/tripletloss/blob/master/02%20-%20tripletloss%20MNIST.ipynb
#         '''
#
#         network = self._buildModel( )
#         # Force the encoding to live on the d-dimentional hypershpere
#         network.add( Lambda( lambda x: K.l2_normalize( x, axis=-1 ) ) )
#         # Define the tensors for the three input images
#         anchor_input = Input( self.input_shape, name="anchor_input" )
#         positive_input = Input( self.input_shape, name="positive_input" )
#         negative_input = Input( self.input_shape, name="negative_input" )
#
#         # Generate the encodings (feature vectors) for the three images
#         encoded_a = network( anchor_input )
#         encoded_p = network( positive_input )
#         encoded_n = network( negative_input )
#         # out = concatenate( [ encoded_a, encoded_p, encoded_n ], axis=1 )
#         # TripletLoss Layer
#         loss_layer = TripletLossLayer( alpha=margin, name='triplet_loss_layer' )( [ encoded_a, encoded_p, encoded_n ] )
#         # Connect the inputs with the outputs
#         self.model = Model( inputs=[ anchor_input, positive_input, negative_input ], outputs=loss_layer )
#         # self.model.add_loss(self.triplet_loss)
#         optimizer = SGD(
#                 lr=self.learning_rate,
#                 # lr_multipliers=learning_rate_multipliers,
#                 momentum=0.5 )
#
#         self.model.compile(loss = None, optimizer=optimizer )
#         # self.model.compile( loss=self.tripletFunc( alpha=margin ), optimizer=optimizer )
#         self.model.summary( )
config = getConfig()
class SiamesWithTriplet:
    def __init__( self,batch_size = config.batch_size,lr = config.lr):
        self.batch_size = batch_size
        self.learning_rate = lr

    def lr_scheduler(self, epoch, lr ):
        decay_rate = 0.1
        decay_step = 12
        if epoch % decay_step == 0 and epoch > 10:
            return lr * decay_rate
        return lr
    def _triplet_loss( self,x ):
        # Triplet Loss function.
        anchor, positive, negative = x
        #        K.l2_normalize
        # distance between the anchor and the positive
        pos_dist = K.sum( K.square( anchor - positive ), axis=1 )
        # distance between the anchor and the negative
        neg_dist = K.sum( K.square( anchor - negative ), axis=1 )

        basic_loss = pos_dist - neg_dist + self.alpha
        loss = K.maximum( basic_loss, 0.0 )
        # print(f'Dp = {pos_dist} Dn = {neg_dist}')
        return loss
    def _identity_loss(self, y_true, y_pred ):
        return K.mean( y_pred )
    def build_embedding_network(self,mode:str='1D' ):
        if mode == '1D':
            network = Sequential( )
            network.add( Conv1D( filters=512, input_shape=([ 1600, 7 ]), activation='relu', kernel_size=10, strides=2,
                            padding='same' ,kernel_regularizer=l2( 1e-3 )) )
            network.add( BatchNormalization( ) )
            network.add( MaxPooling1D( pool_size=3, strides=1 ) )
            # network.add(Dropout( 0.5 ))

            network.add( Conv1D( filters=1024, activation='relu', kernel_size=5, padding='same',kernel_regularizer=l2( 1e-3 ) ) )
            network.add( BatchNormalization( ) )
            network.add( MaxPooling1D( pool_size=3, strides=1 ) )
            # network.add( Dropout( 0.5 ) )

            network.add( Conv1D( filters=512, activation='relu', kernel_size=5, padding='same',kernel_regularizer=l2( 1e-3 ) ) )
            network.add( Conv1D( filters=256, activation='relu', kernel_size=10, padding='same',kernel_regularizer=l2( 1e-2 ) ) )
            # network.add( MaxPooling1D( pool_size=4, strides=2 ) )
            network.add( BatchNormalization( ) )
            network.add( Flatten( ) )
            # network.add( Dropout( 0.5 ) )

            network.add( Lambda( lambda x: K.l2_normalize( x, axis=-1 )) )
            self.embedding_network = network
        elif mode == '2D':
            network = Sequential()
            network.add(Conv2D(filters = 32,kernel_size = 4,activation='relu', input_shape=[200,60,3],padding = 'valid'))
            network.add( BatchNormalization( ) )
            network.add(MaxPooling2D(pool_size=4,strides = 4,padding='valid',name = 'feature_layer'))

            # network.add(Conv2D( filters=64, kernel_size=4, activation='relu', padding='valid' ) )
            # network.add( BatchNormalization( ) )
            # network.add( MaxPooling2D( pool_size=4, strides=4, padding='valid' ) )

            # network.add( Conv2D( filters=32, kernel_size=4, activation='relu', padding='valid' ) )
            # network.add( BatchNormalization( ) )
            # network.add( MaxPooling2D( pool_size=4, strides=4, padding='valid' ) )

            network.add( Flatten( ) )

            network.add( Lambda( lambda x: K.l2_normalize( x, axis=-1 ) ) )
            # optimizer = tf.keras.optimizers.SGD(
            #         learning_rate=0.01, momentum=0.9
            # )
            # network.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
            # network.summary( )
        elif mode == 'signFi':
            network = Sequential( )
            network.add(
                Conv2D( filters=4, kernel_size=4, activation='relu', input_shape=[ 200, 60, 3 ], padding='valid' ) )
            network.add( BatchNormalization( ) )
            network.add( MaxPooling2D( pool_size=4, strides=4, padding='valid' ) )
            # network.add(Dropout(0.1))
            network.add( Flatten( ) )
            network.add( Dense( units=config.num_classes, activity_regularizer=l2( 1e-3 ) ) )
            # network.add( Dense( units=552, ) )
            network.add( Softmax( ) )
            network.add( Lambda( lambda x: K.l2_normalize( x, axis=-1 ) ) )
            optimizer = tf.keras.optimizers.SGD(
                    learning_rate=0.01, momentum=0.9
            )
            network.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
            network.summary( )
        return network
    def build_TripletModel( self, network,data_dir,margin ):
        '''
        Define the Keras Model for training
            Input :
                input_shape : shape of input images
                ten_ges_embedding_network : Neural ten_ges_embedding_network to train outputing embeddings
                margin : minimal distance between Anchor-Positive and Anchor-Negative for the loss function (alpha)
            https://github.com/pranjalg2308/siamese_triplet_loss/blob/master/Siamese_With_Triplet_Loss.ipynb
            https://keras.io/examples/vision/siamese_network/
            https://github.com/CrimyTheBold/tripletloss/blob/master/02%20-%20tripletloss%20MNIST.ipynb
        '''
        self.gestureDataLoader = gestureDataLoader(
                data_path=data_dir, batch_size=self.batch_size )
        # self.summary_writer = tf.summary.create_file_writer( tensorboard_log_path )
        self.input_shape = self.gestureDataLoader.InputShape
        self.alpha = margin
        # Define the tensors for the three input images
        anchor_input = Input( self.input_shape, name="anchor_input" )
        positive_input = Input( self.input_shape, name="positive_input" )
        negative_input = Input( self.input_shape, name="negative_input" )

        # Generate the encodings (feature vectors) for the three images
        encoded_a = network( anchor_input )
        encoded_p = network( positive_input )
        encoded_n = network( negative_input )

        loss = Lambda( self.triplet_loss )( [ encoded_a, encoded_p, encoded_n ] )
        self.model = Model( inputs=[ anchor_input, positive_input, negative_input ], outputs=loss )
        # self.model.add_loss(CustomMSE())
        optimizer = SGD(
                lr=self.learning_rate,
                # lr_multipliers=learning_rate_multipliers,
                momentum=0.5 )
        self.model.compile(loss=self.identity_loss,optimizer=optimizer,metrics = None )
        self.model.summary( )
        return self.model
def OneShotTesting( test_dir:str,embedding_model ):
    test_sample = 100
    nway_min = 2
    nway_max = 6
    test_acc = [ ]
    nway_list = [ ]

    for nway in range( nway_min, nway_max + 1 ):
        print( "Checking %d way accuracy...." % nway )
        correct_count = 0
        for _ in range( test_sample ):
            # Retrieving nway number of triplets and calculating embedding vector
            nway_anchor, nway_positive, _ = gestureDataLoader( data_path=test_dir,
                                                               batch_size=nway ).getTripletTrainBatcher( )
            # support set, it has N different classes depending on the batch_size
            # nway_anchor has the same class with nway_positive at the same row
            nway_anchor_embedding = embedding_model.predict( nway_anchor )

            sample_index = random.randint( 0, nway - 1 )
            sample_embedding = embedding_model.predict( np.expand_dims( nway_positive[ sample_index ], axis=0 ) )
            # print(sample_index, nway_anchor_embedding.shape, sample_embedding.shape)
            distance = K.sum( K.square( nway_anchor_embedding - sample_embedding ), axis = 1 )
            if np.argmin( distance ) == sample_index:
                correct_count += 1
        nway_list.append( nway )
        acc = (correct_count / test_sample) * 100.
        test_acc.append( acc )
        print( "Accuracy %.2f" % acc )
class OneShotCallback(tf.keras.callbacks.Callback):
    def passNetworks(self,network):
        self.network = network
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 30 == 1 and epoch > 10:
            OneShotTesting( test_dir='./20181115/', embedding_model=self.network )
if __name__ == '__main__':
    '''Prepare for training'''
    network_train = SiamesNetworkTriplet_2(batch_size=1000,lr=0.001,margin = 2.7,data_dir = './20181116/')
    ten_ges_embedding_network = network_train.build_embedding_network( )
    model = network_train.build_TripletModel( network = ten_ges_embedding_network )

    callbacks = OneShotCallback()
    callbacks.passNetworks( network = ten_ges_embedding_network )
    dataGenerator = network_train.gestureDataLoader.tripletsDataGenerator()

    history = model.fit( dataGenerator,
                         epochs = 1,
                         steps_per_epoch=10,
                         verbose=True,
                         callbacks=[ callbacks,tf.keras.callbacks.LearningRateScheduler(network_train.lr_scheduler, verbose=1) ]
                         )
    OneShotTesting( test_dir = './20181115/',
                    embedding_model=ten_ges_embedding_network,
                    )
    if not os.path.exists( './models' ):
        os.makedirs( './models' )
    ten_ges_embedding_network.save('Triplet_loss_model.h5')
    # ten_ges_embedding_network = SiamesNetworkTriplet( batch_size=32, data_dir='./20181115/' )
    # acc = ten_ges_embedding_network.train_siamese_network(number_of_iterations = 1000,evaluate_each = 1000,model_name='testing')
    
