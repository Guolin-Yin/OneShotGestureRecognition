import numpy as np
import tensorflow as tf
class getConfig:
    def __init__(self):
        self.train_dir = 'D:/Matlab/SignFi/Dataset'
        self.eval_dir = None
        self.lr = 1e-2
        self.batch_size = 32
        self.N_train_classes = None
        self.num_finetune_classes = None
        self.source = None
        self.input_shape = [200,60,3]
        self.pretrainedfeatureExtractor_path = None
        self.tunedModel_path = None
        self.record = None
        self.weight = None
        self.weight2 = None
        self.initGPU()
    def setSavePath(self,val_acc):
        self.feature_extractor_save_path = f'./models/feature_extractor_train_on_user-' \
                                           f'{self.source[0]}-{self.source[ 1 ]}-{self.source[ 2 ]}-{self.source[ 3 ]}-on-' \
                                           f'{self.N_train_classes}-classes-val_acc-{val_acc}.h5'
        self.fine_Tune_model_save_path = f'./models/fine_Tune_model_one_shot_on_user-{self.source[4]}.h5'
    def modelDictionary( self ):
        '''
        pretrained model:
        * Trained on lab environment, 125 classes, user 1 to 4:
            Path: ./models/signFi_wholeModel_weight_AlexNet_training_acc_0.90_on_125cls_user1to4.h5
            * Fine Tuned Model, on user 5:
            Path: './models/fc_fineTuned_one_shot_with_Zscore.h5' -> 87.8%
                './models/fc_fineTuned_two_shot_with_Zscore.h5' -> 94.2%
                './models/fc_fineTuned_3_shot_with_Zscore.h5' -> 94.7%
                './models/fc_fineTuned_4_shot_with_Zscore.h5' -> 98.1%
                './models/fc_fineTuned_five_shot_with_Zscore.h5' -> 98%
        * Train on Lab environment, 250 classes, user 5
            './models/signFi_featureExtractor_weight_AlexNet_lab_training_acc_0.95_on_250cls.h5'
            './models/feature_extractor_weight_Alexnet_lab_250cls_val_acc_0.94_with_Zscore.h5'
            './models/feature_extractor_weight_Alexnet_home_250cls_val_acc_0.97_with_Zscore.h5'
        :return:
        '''
        dict = {'feature_extractor_lab125user1234':'./models/signFi_wholeModel_weight_AlexNet_training_acc_0'
                                                   '.90_on_125cls_user1to4.h5',
                'fineTunedModel_lab125user1234_onUser5':'./models/fc_fineTuned_one_shot.h5',
                'feature_extractor_lab250user5':'./models/signFi_featureExtractor_weight_AlexNet_training_acc_0' \
                                                '.95_on_250cls.h5'
                }
        return dict
    def best(self):
        '''
        One shot tuning-20181115: 0.518
        Best record:
        [5,14,14,14,0,18]
        Two shots tuning-20181115: 69.00
        Best record:
                     [array([ 7, 15]),
                     array([15,  2]),
                     array([ 1, 13]),
                     array([19,  4]),
                     array([18, 17]),
                     array([ 3, 17])]
         Three shot tuning-20181115:
         # 78.4 record:[[16,  2, 13],
         #             [15, 13,  4],
         #             [ 5,  1, 18],
         #             [ 6, 10,  4],
         #             [5, 3, 9],
         #             [15,  7, 17]]
         Best record:[[ 2,  8, 17],
                     [ 8, 11, 15],
                     [11,  3, 10],
                     [19,  4, 12],
                     [11,  8,  5],
                     [19,  5, 15]]
         seven shot tuning-20181115:
         Best record:[[9, 12, 2, 13, 17, 16, 6],
                     [5, 8, 17, 11, 18, 10, 4],
                     [13, 8, 10, 4, 9, 5, 14],
                     [18, 14, 11, 12, 5, 3, 2],
                     [11, 9, 10, 15, 0, 2, 17],
                     [2, 18, 10, 15, 1, 16, 9]]
        '''
        pass
    def initGPU(self):
        gpus = tf.config.experimental.list_physical_devices( 'GPU' )
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth( gpu, True )
            except RuntimeError as e:
                print( e )