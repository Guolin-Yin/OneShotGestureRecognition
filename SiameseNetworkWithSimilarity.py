import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
from Preprocess.gestureDataLoader import gestureDataLoader
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import numpy as np

class TripletLossLayer( Layer ):
    def __init__( self, alpha, **kwargs ):
        self.alpha = alpha
        super( TripletLossLayer, self ).__init__( **kwargs )

    def triplet_loss( self, inputs ):
        anchor, positive, negative = inputs
        p_dist = K.sum( K.square( anchor - positive ), axis=-1 )
        n_dist = K.sum( K.square( anchor - negative ), axis=-1 )
        return K.sum( K.maximum( p_dist - n_dist + self.alpha, 0 ), axis=0 )

    def call( self, inputs ):
        loss = self.triplet_loss( inputs )
        self.add_loss( loss )
        return loss
class SiamesNetwork:
    def __init__(self,data_path, learning_rate,batch_size,
                 l2_regularization_penalization, tensorboard_log_path):
        self.input_shape = [90,1600,1]
        self.model = []
        self.learning_rate = learning_rate
        self.gestureDataLoader = gestureDataLoader(
                data_path=data_path, batch_size=batch_size )
        self.summary_writer = tf.summary.create_file_writer( tensorboard_log_path )
        self._construct_siamese_architecture(
                # learning_rate_multipliers,
                                              l2_regularization_penalization)
    def _construct_siamese_architecture(self,
                                        # learning_rate_multipliers,
                                         l2_regularization_penalization):
        convolutional_net = Sequential( )
        convolutional_net.add( Conv2D( filters=64, kernel_size=(10, 20),
                                       activation='relu',
                                       input_shape=self.input_shape,
                                       strides = (2,2),
                                       kernel_regularizer=l2(
                                               l2_regularization_penalization[ 'Conv1' ] ),
                                       name='Conv1' ) )
        convolutional_net.add( MaxPool2D( ) )

        convolutional_net.add( Conv2D( filters=128, kernel_size=(7, 7),
                                       activation='relu',
                                       kernel_regularizer=l2(
                                               l2_regularization_penalization[ 'Conv2' ] ),
                                       name='Conv2' ) )
        convolutional_net.add( MaxPool2D( ) )

        convolutional_net.add( Conv2D( filters=128, kernel_size=(4, 4),
                                       activation='relu',
                                       kernel_regularizer=l2(
                                               l2_regularization_penalization[ 'Conv3' ] ),
                                       name='Conv3' ) )
        # convolutional_net.add( MaxPool2D( ) )
        #
        # convolutional_net.add( Conv2D( filters=256, kernel_size=(4, 4),
        #                                activation='relu',
        #                                kernel_regularizer=l2(
        #                                        l2_regularization_penalization[ 'Conv4' ] ),
        #                                name='Conv4' ) )

        convolutional_net.add( Flatten( ) )
        convolutional_net.add( Dense( units=128,
                                      activation='sigmoid',
                                      kernel_regularizer=l2( l2_regularization_penalization[ 'Dense1' ] ),
                                      name='Dense1' ) )

        # Now the pairs of images
        input_image_1 = Input( self.input_shape )
        input_image_2 = Input( self.input_shape )

        encoded_image_1 = convolutional_net( input_image_1 )
        encoded_image_2 = convolutional_net( input_image_2 )

        # L1 distance layer between the two encoded outputs
        # One could use Subtract from Keras, but we want the absolute value
        l1_distance_layer = Lambda( lambda tensors: K.abs( tensors[ 0 ] - tensors[ 1 ] ) )
        l1_distance = l1_distance_layer( [ encoded_image_1, encoded_image_2 ] )

        # Same class or not prediction
        prediction = Dense( units=1, activation='sigmoid' )( l1_distance )
        self.model = Model( inputs=[ input_image_1, input_image_2 ], outputs=prediction )

        # Define the optimizer and compile the model
        optimizer = SGD(
                lr=self.learning_rate,
                # lr_multipliers=learning_rate_multipliers,
                momentum=0.5 )

        self.model.compile( loss='binary_crossentropy', metrics=[ 'binary_accuracy' ],
                            optimizer=optimizer )
        convolutional_net.summary( )
    def _write_logs_to_tensorboard(self, current_iteration, train_losses,
                                    train_accuracies, validation_accuracy,
                                    evaluate_each):
        for index in range( 0, evaluate_each ):
            with self.summary_writer.as_default( ):
                tf.summary.scalar( 'Train loss', train_losses[index], step=index )
                tf.summary.scalar( 'Train accuracy', train_accuracies[index], step=index )

                if index == (evaluate_each - 1):
                    self.step += int((index+1)/1000)
                    tf.summary.scalar( 'One-Shot Validation Accuracy', validation_accuracy, step=self.step )
    def train_siamese_network(self,
                              number_of_iterations,
                              support_set_size,
                              final_momentum,
                              momentum_slope,
                              evaluate_each,
                              model_name):
        print( "Num GPUs Available: ", len( tf.config.experimental.list_physical_devices( 'GPU' ) ) )
        train_losses = np.zeros( shape=(evaluate_each) )
        train_accuracies = np.zeros( shape=(evaluate_each) )
        # training params
        trainCount = 0
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0

        for iter in range(number_of_iterations):
            data,labels = self.gestureDataLoader.getTrainBatcher()
            train_loss, train_acc = self.model.train_on_batch(data,labels)
            # Update learning rate and momentum
            if (iter + 1)% 500 == 0:
                K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr)*0.99)
            if K.get_value(self.model.optimizer.momentum) < final_momentum:
                K.set_value( self.model.optimizer.momentum,
                             K.get_value( self.model.optimizer.momentum ) + momentum_slope )

            train_losses[trainCount] = train_loss
            train_accuracies[trainCount] = train_acc
            trainCount += 1
            print( 'Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                   (iter + 1, number_of_iterations, train_loss, train_acc, K.get_value(
                           self.model.optimizer.lr )) )
            # perform a one shot task evaluation on the validation data
            if (iter + 1) % evaluate_each == 0:
                num_of_run_per_gesture = 40
                validation_accuracy = self.gestureDataLoader.oneShotTest(self.model,
                                                                         support_set_size,
                                                                         number_of_run_per_gesture,
                                                                         is_validation=True)
                self._write_logs_to_tensorboard(
                        iteration, train_losses, train_accuracies,
                        validation_accuracy, evaluate_each )
                trainCount = 0
                if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' +
                          str(best_validation_accuracy))
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iter

                        model_json = self.model.to_json()

                        if not os.path.exists("./models"):
                            os.makedirs("./models")
                        with open("models/"+model_name+ ".json","w") as json_file:
                            json_file.write(model_json)
                        self.model.save_weights("models/"+model_name+".h5")
                    if iter - best_accuracy_iteration > 1000:
                        print(
                                'Early Stopping: validation accuracy did not increase for 10000 iterations' )
                        print( 'Best Validation Accuracy = ' +
                               str( best_validation_accuracy ) )
                        print( 'Validation Accuracy = ' + str( best_validation_accuracy ) )
                        break
        print("The end of training process")
        return best_validation_accuracy

    # def triplet_loss( self,inputs, dist='sqeuclidean', margin='maxplus' ):
    #     anchor, positive, negative = inputs
    #     positive_distance = K.square( anchor - positive )
    #     negative_distance = K.square( anchor - negative )
    #     if dist == 'euclidean':
    #         positive_distance = K.sqrt( K.sum( positive_distance, axis=-1, keepdims=True ) )
    #         negative_distance = K.sqrt( K.sum( negative_distance, axis=-1, keepdims=True ) )
    #     elif dist == 'sqeuclidean':
    #         positive_distance = K.sum( positive_distance, axis=-1, keepdims=True )
    #         negative_distance = K.sum( negative_distance, axis=-1, keepdims=True )
    #     loss = positive_distance - negative_distance
    #     if margin == 'maxplus':
    #         loss = K.maximum( 0.0, 1 + loss )
    #     elif margin == 'softplus':
    #         loss = K.log( 1 + K.exp( loss ) )
    #     return K.mean( loss )
    # def get_embedding_model(self,  input_shape, embedding_dim ):
    #     _input = Input( shape=input_shape )
    #     x = Flatten( )( _input )
    #     x = Dense( embedding_dim * 4, activation="relu" )( x )
    #     x = Dense( embedding_dim * 2, activation='relu' )( x )
    #     x = Dense( embedding_dim )( x )
    #     return Model( _input, x )
    # def get_siamese_model( self, input_shape, triplet_margin=.3, embedding_dim=50 ):
    #     """
    #         Model architecture
    #     """
    #
    #     # Define the tensors for the triplet of input images
    #     anchor_input = Input( input_shape, name="anchor_input" )
    #     positive_input = Input( input_shape, name="positive_input" )
    #     negative_input = Input( input_shape, name="negative_input" )
    #
    #     # Convolutional Neural Network (same from earlier)
    #     embedding_model = get_embedding_model( input_shape, embedding_dim )
    #
    #     # Generate the embedding outputs
    #     encoded_anchor = embedding_model( anchor_input )
    #     encoded_positive = embedding_model( positive_input )
    #     encoded_negative = embedding_model( negative_input )
    #
    #     inputs = [ anchor_input, positive_input, negative_input ]
    #     outputs = [ encoded_anchor, encoded_positive, encoded_negative ]
    #
    #     # Connect the inputs with the outputs
    #     siamese_triplet = Model( inputs=inputs, outputs=outputs )
    #
    #     siamese_triplet.add_loss( (self.triplet_loss( outputs, dist='euclidean', margin='maxplus' )) )
    #
    #     # return the model
    #     return embedding_model, siamese_triplet

    # L2 Distance
def trainSiameseSimMain( ):
    dataset_path = './20181115/'
    use_augmentation = True
    learning_rate = 10e-4
    batch_size = 32
    # Learning Rate multipliers for each layer
    learning_rate_multipliers = { }
    learning_rate_multipliers[ 'Conv1' ] = 1
    learning_rate_multipliers[ 'Conv2' ] = 1
    learning_rate_multipliers[ 'Conv3' ] = 1
    learning_rate_multipliers[ 'Conv4' ] = 1
    learning_rate_multipliers[ 'Dense1' ] = 1
    # l2-regularization penalization for each layer
    l2_penalization = { }
    l2_penalization[ 'Conv1' ] = 1e-2
    l2_penalization[ 'Conv2' ] = 1e-2
    l2_penalization[ 'Conv3' ] = 1e-2
    l2_penalization[ 'Conv4' ] = 1e-2
    l2_penalization[ 'Dense1' ] = 1e-4
    # Path where the logs will be saved
    tensorboard_log_path = './logs/siamese_net_lr10e-4'
    siamese_network = SiamesNetwork(
            data_path=dataset_path,
            learning_rate=learning_rate,
            batch_size=batch_size,
            # learning_rate_multipliers=learning_rate_multipliers,
            l2_regularization_penalization=l2_penalization,
            tensorboard_log_path=tensorboard_log_path
    )
    # Final layer-wise momentum (mu_j in the paper)
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    support_set_size = 20
    evaluate_each = 1000
    number_of_train_iterations = 1000000

    validation_accuracy = siamese_network.train_siamese_network( number_of_iterations=number_of_train_iterations,
                                                                 support_set_size=support_set_size,
                                                                 final_momentum=momentum,
                                                                 momentum_slope=momentum_slope,
                                                                 evaluate_each=evaluate_each,
                                                                 model_name='siamese_net_lr10e-4' )
    if validation_accuracy == 0:
        evaluation_accuracy = 0
    else:
        # Load the weights with best validation accuracy
        siamese_network.model.load_weights( './models/siamese_net_lr10e-4.h5' )
        evaluation_accuracy = siamese_network.omniglot_loader.one_shot_test( siamese_network.model,
                                                                             20, 40, False )

    print( 'Final Evaluation Accuracy = ' + str( evaluation_accuracy ) )
if __name__ == '__main__':
    # gestureDataLoader = gestureDataLoader( )
    # data, label = gestureDataLoader.getTrainBatcher( )
    # triplets = gestureDataLoader.getTripletTrainBatcher( )
    network = SiamesNetworkTriplet(batch_size=32,data_dir = './20181115/')
    network.train_siamese_network(number_of_iterations = 10000, evaluate_each =500, model_name = 'siameseTriplet')
    # trainSiameseSimMain( )