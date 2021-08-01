import numpy as np
import tensorflow as tf
class getConfig:
    def __init__(self):
        self.train_dir = 'D:/Matlab/SignFi/Dataset'
                            # 'E:/Widar_dataset_matfiles/20181109/User1',
                            # 'E:/Widar_dataset_matfiles/20181112/User1',
                            # 'E:/Widar_dataset_matfiles/20181109/User2',
                            # 'E:/Widar_dataset_matfiles/20181112/User2',
                            # 'E:/Widar_dataset_matfiles/20181115'

        self.eval_dir = 'E:/Widar_dataset_matfiles/20181121/test_user1_r3'
        self.lr = 1e-2
        self.N_train_classes = 125
        self.batch_size = 32
        self.input_shape = [200,60,3]
        self.featureExtractor_path = './models/signFi_wholeModel_weight_AlexNet_training_acc_0.89_on_125cls_user1to4.h5'
        self.tunedModel_path = './models/fc_fineTuned_one_shot_with_Zscore.h5'
        self.initGPU()
    def get_params(self):
        return {train_dir: self.train_dir,
                eval_dir: self.eval_dir}
    def initGPU(self):
        gpus = tf.config.experimental.list_physical_devices( 'GPU' )
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth( gpu, True )
            except RuntimeError as e:
                print( e )
    def modelDictionary( self ):
        '''
        pretrained model:
        * Trained on lab environment, 125 classes, user 1 to 4:
            Path: ./models/signFi_wholeModel_weight_AlexNet_training_acc_0.90_on_125cls_user1to4.h5
            with Z-score:
            './models/fc_fineTuned_one_shot_with_Zscore.h5' -> 87.8%
            * Fine Tuned Model, on user 5:
            Path: './models/fc_fineTuned_one_shot.h5'
        * Train on Lab environment, 250 classes, user 5
            ./models/signFi_featureExtractor_weight_AlexNet_training_acc_0.95_on_250cls.h5
        :return:
        '''
        dict = {'feature_extractor_lab125user1234':'./models/signFi_wholeModel_weight_AlexNet_training_acc_0'
                                                   '.90_on_125cls_user1to4.h5',
                'fineTunedModel_lab125user1234_onUser5':'./models/fc_fineTuned_one_shot.h5',
                'feature_extractor_lab250user5':'./models/signFi_featureExtractor_weight_AlexNet_training_acc_0' \
                                                '.95_on_250cls.h5'
                }
        return dict