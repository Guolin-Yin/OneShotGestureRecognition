from tensorflow import keras
from keras.layers import Input, Lambda, Dropout, ReLU, Add, ELU, LeakyReLU, ELU, Conv1D
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import (Dense, Conv2D, Flatten,BatchNormalization,Activation,
                          MaxPooling2D, LSTM, TimeDistributed, AveragePooling2D)

import numpy as np

import signal_representations as sr
from augmentation import  awgn, channel_aug

# In[]
'''Residual block'''

# def resblock(x, kernelsize, filters, downsample = False):
#     fx = Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
#     # fx = BatchNormalization()(fx)
#     fx = Conv2D(filters, kernelsize, padding='same')(fx)
    
    
#     if downsample:
#         shortcut = Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
#         out = Add()([shortcut,fx])
#         out = ReLU()(out)
#     else:
#         out = Add()([x,fx])
#         out = ReLU()(out)
#     # out = keras.layers.BatchNormalization()(out)
#     return out 

def resblock(x, kernelsize, filters, first_layer = False):

    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        x = Conv2D(filters, 1, padding='same')(x)
        
        out = Add()([x,fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        
        out = Add()([x,fx])
        out = ReLU()(out)

    return out 


# In[]  
'''Class triplet ten_ges_embedding_network'''
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)           

    
class Triplet():
    def __init__(self):
        pass
        
    def create_triplet_net(self, embedding_net, alpha):
        
#        embedding_net = encoder()
        self.alpha = alpha
        
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_3 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N = embedding_net(input_3)
   
        loss = Lambda(self.triplet_loss)([A, P, N]) 
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        return model
      
    def triplet_loss(self,x):
    # Triplet Loss function.
        anchor,positive,negative = x
#        K.l2_normalize
    # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative),axis=1)

        basic_loss = pos_dist-neg_dist + self.alpha
        loss = K.maximum(basic_loss,0.0)
        return loss   
    
    def encoder(self, datashape):
            
        self.datashape = datashape
        
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        
        # x = MaxPooling2D(pool_size=(3,3),strides = 2)(x)
        
        x = resblock(x, 3, 32)
        x = resblock(x, 3, 32)
        # x = resblock(x, 3, 32)
        
        x = resblock(x, 3, 64, first_layer = True)
        x = resblock(x, 3, 64)
        # x = resblock(x, 3, 64)
        
        # x = resblock(x, 3, 128, first_layer = True)
        # x = resblock(x, 3, 128)
        # x = resblock(x, 3, 256)
        
        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512)(x)
        # x = ReLU()(x)
        # x = ELU()(x)
        
        
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model             
    
    def encoder2(self, datashape):
        
        self.datashape = datashape
        
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        
        x = Conv2D(50, (1,7), activation='relu', padding='same')(inputs)
        
        # x = MaxPooling2D(pool_size=(1,4))(x)
        
        x = Conv2D(50, (1,7), activation='relu', padding='same')(x)
        
        x = MaxPooling2D(pool_size=(1,4))(x)
        
        # x = Conv2D(32, (2,128), activation='relu', padding='same')(x)
        # x = MaxPooling2D(pool_size=(3,3),strides = 2)(x)
        
        # x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512)(x)
        # x = ReLU()(x)
        x = ELU()(x)
        
        
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_triplet(self):
        """Choose a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label and
        anchor and negative have different labels."""
        
        
        n = a = self.dev_range[np.random.randint(len(self.dev_range))]
        
        while n == a:
            # keep searching randomly!
            n = self.dev_range[np.random.randint(len(self.dev_range))]
        a, p = self.call_sample(a), self.call_sample(a)
        n = self.call_sample(n)
        
        return a, p, n

    def call_sample(self,label_name):
        """Choose an image from our training or test data with the
        given label."""
        num_sample = len(self.label)
        idx = np.random.randint(num_sample)
        while self.label[idx] != label_name:
            # keep searching randomly!
            idx = np.random.randint(num_sample) 
        return self.data[idx]

    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        self.dev_range = dev_range
        
        # center_freq = 868.1e6
        # fs = 1e6
        # speed_range = np.arange(0,3)   # define speed range
        # rms_delay_range = np.arange(0,400)   # define rms delay spread range
        # snr_range = np.arange(30,70)  # define SNR range
        
        while True:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batchsize):
                a, p, n = self.get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            
            # A = np.array(list_a, dtype='complex')
            # P = np.array(list_p, dtype='complex')
            # N = np.array(list_n, dtype='complex')
            
            # batch_list = []
            # batch_list.extend(list_a)
            # batch_list.extend(list_p)
            # batch_list.extend(list_n)
            
            # triplet_batch = np.array(batch_list, dtype='complex')
            
            # triplet_batch = np.concatenate([A,P,N])
            
            # triplet_batch = channel_aug(triplet_batch, center_freq, fs, 
            #                               speed_range,
            #                               rms_delay_range)

            # triplet_batch = awgn(triplet_batch,snr_range)
            
            # transform to differential spectrograms 
            # triplet_batch = sr.dspectrogram(triplet_batch)
            
            # A, P, N = np.array_split(triplet_batch,3)
            
           # a "dummy" label which will come in to our identity loss
           # function below as y_true. We'll ignore it.
            label = np.ones(batchsize)
            yield [A, P, N], label  

# In[]
'''Class model ten_ges_embedding_network'''
def contrastive_loss(y_true, y_pred):

    margin = 0.01
    # square_pred = K.square(y_pred)
    # margin_square = K.square(K.maximum(margin - y_pred, 0))
    #return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

    
class Siamese():
    def __init__(self):
        pass
        
    def create_siamese_net(self, embedding_net):
        
#        embedding_net = encoder()
        
        input_1 = Input(np.append(self.datashape[1:-1],1))
        input_2 = Input(np.append(self.datashape[1:-1],1))
        
        left_out = embedding_net(input_1)
        right_out = embedding_net(input_2)
   
        distance = Lambda(self.eu_distance)([left_out,right_out]) 
        model = Model(inputs=[input_1, input_2], outputs = distance)
        return model
      
    
    def eu_distance(self,vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    
    
    def encoder(self, datashape):
            
        self.datashape = datashape
        
        inputs = Input(shape=(np.append(self.datashape[1:-1],1)))
        
        x = Conv2D(64, 7, strides = (2,2), activation='relu', padding='same')(inputs)
        
        x = MaxPooling2D(pool_size=(2,2))(x)
        
        x = resblock(x, 3, 64)
        x = resblock(x, 3, 64)
        # x = resblock(x, 3, 32)
        
        x = resblock(x, 3, 128, with_conv_shortcut = True)
        x = resblock(x, 3, 128)
        # x = resblock(x, 3, 64)
        
        # x = resblock(x, 3, 256, with_conv_shortcut = True)
        # x = resblock(x, 3, 256)
        
        x = AveragePooling2D(pool_size=(2,2))(x)
        
        x = Flatten()(x)
    
        x = Dense(512)(x)
        # x = ReLU()(x)
        x = ELU()(x)
        
        
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model     
    
    def get_neg_pairs(self):
        """Choose a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label and
        anchor and negative have different labels."""

        n = p = np.random.randint(self.dev_range[0],self.dev_range[-1])
        while n == p:
            # keep searching randomly!
            n = np.random.randint(self.dev_range[0],self.dev_range[-1])
        p = self.call_sample(p)
        n = self.call_sample(n)
        return p, n
    
    def get_pos_pairs(self):
        """Choose a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label and
        anchor and negative have different labels."""

        dev_label = np.random.randint(self.dev_range[0],self.dev_range[-1])
        
        p1 = self.call_sample(dev_label)
        p2 = self.call_sample(dev_label)
        return p1,p2
            
    def call_sample(self,label_name):
        """Choose an image from our training or test data with the
        given label."""
        num_sample = len(self.label)
        idx = np.random.randint(num_sample)
        while self.label[idx] != label_name:
            # keep searching randomly!
            idx = np.random.randint(num_sample) 
        return self.data[idx]


    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        self.dev_range = dev_range

        while True:
            list_left = []
            list_right = []
            
            label = np.zeros(batchsize)
            label[batchsize//2:] = 1    # 1 indicates negative pairs
            
            for i in range(batchsize):

                if i >= batchsize // 2:
                    left_input, right_input = self.get_neg_pairs()
                    list_left.append(left_input)
                    list_right.append(right_input)
                else: 
                    left_input, right_input = self.get_pos_pairs()
                    list_left.append(left_input)
                    list_right.append(right_input)

            
            L = np.array(list_left, dtype='float32')
            R = np.array(list_right, dtype='float32')
            # a "dummy" label which will come in to our identity loss
            # function below as y_true. We'll ignore it.
            
            yield [L, R], label    


# In[] 
'''Classification ten_ges_embedding_network'''


def classification_net(datashape, num_classes):
    
    datashape = datashape
    
    inputs = Input(shape=(np.append(datashape[1:-1],1)))
    
    x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
    
    # x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    # x = resblock(x, 3, 64)
    
    x = resblock(x, 3, 64, first_layer = True)
    x = resblock(x, 3, 64)
    # x = resblock(x, 3, 128)
    
    # x = resblock(x, 3, 256, with_conv_shortcut = True)
    # x = resblock(x, 3, 256)
    
    x = AveragePooling2D(pool_size=2)(x)
    
    x = Flatten()(x)

    x = Dense(512)(x)
    # x = LeakyReLU()(x)
    x = ELU()(x)
    
    x = Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)
    
    # x =  Dense(256, activation= 'relu')(x)
    
    outputs = Dense(num_classes, activation= 'softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model 

def remove_dense(model):
    encoder = Model(inputs=model.input, outputs= model.get_layer('feature_layer').output)
    return encoder


def compute_probs( network, X, Y ):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class

    Returns
        probs : array of shape (m,m) containing distances

    '''
    m = X.shape[ 0 ]
    nbevaluation = int( m * (m - 1) / 2 )
    probs = np.zeros( (nbevaluation) )
    y = np.zeros( (nbevaluation) )

    # Compute all embeddings for all pics with current network
    embeddings = network.predict( X )

    size_embedding = embeddings.shape[ 1 ]

    # For each pics of our dataset
    k = 0
    for i in range( m ):
        # Against all other images
        for j in range( i + 1, m ):
            # compute the probability of being the right decision : it should be 1 for right class, 0 for all other
            # classes
            probs[ k ] = -compute_dist( embeddings[ i, : ], embeddings[ j, : ] )
            if (Y[ i ] == Y[ j ]):
                y[ k ] = 1
                # print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
            else:
                y[ k ] = 0
                # print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
            k += 1
    return probs, y


def compute_metrics( probs, yprobs ):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with
        score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score
        >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0]
        represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score( yprobs, probs )
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve( yprobs, probs )

    return fpr, tpr, thresholds, auc
