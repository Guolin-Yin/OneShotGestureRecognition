from keras.models import load_model
from SiameseNetworkWithTripletLoss import *
def continue_training():
    encoder = load_model( './models/Triplet_loss_model.h5', compile=False )
    network_train = SiamesNetworkTriplet_2( batch_size=32, data_dir='./20181116/' )
    model = network_train.build_TripletModel( network=encoder )

    callbacks = OneShotCallback()
    callbacks.passNetworks( network = ten_ges_embedding_network )
    dataGenerator = network_train.gestureDataLoader.tripletsDataGenerator()

    history = model.fit( dataGenerator,
                         epochs = 1000,
                         steps_per_epoch=100,
                         verbose=True,
                         callbacks=[ callbacks ]
                         )
