import numpy as np
import tensorflow as tf
from scipy.io import loadmat
class getConfig:
    def __init__(self,if_Restore_Samp_idx:bool = False):
        self.epoch =  150
        self.train_dir = None
        self.eval_dir = None
        self.lr = 1e-3
        self.batch_size = 32
        # To split the dataset into training and testing
        self.N_base_classes = None
        # To define the fine tuning model
        self.N_novel_classes = None
        self.source = None
        self.input_shape = [200,60,3]
        self.pretrainedfeatureExtractor_path = None
        self.tunedModel_path = None
        self.record = None
        self.nshots = None
        self.nshots_per_domain = None
        self.domain_selection = None
        self.nways = None
        self.matPath = None
        self.test_Domain = (2,3,3)
        self.learning_Domain = [(2,1,3), (2,2,3), (2,3,3), (2,4,3), (2,5,3)]
        self.history = None
        self.n_ft_cls = None
        # if if_Restore_Samp_idx:
        #     self.getSampleIdx()
        self.initGPU()
    def setSavePath(self,val_acc):
        self.feature_extractor_save_path = f'./models/feature_extractor_train_on_user-' \
                                           f'{self.source[0]}-{self.source[ 1 ]}-{self.source[ 2 ]}-{self.source[ 3 ]}-on-' \
                                           f'{self.N_base_classes}-classes-val_acc-{val_acc}.h5'
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
    def setMatSavePath(self, path):
        self.matPath = path
    def getSampleIdx(self):
        '''
        One shot tuning-20181115: 0.518
        Best record:
        [5,14,14,14,0,18]
        Two shots tuning-20181115: 69.00
        Best record:
                     [[ 7, 15],
                     [15,  2],
                     [ 1, 13],
                    [19,  4],
                     [18, 17],
                     [ 3, 17]]
         Three shot tuning-20181115:

         Best record:[[ 2,  8, 17],
                     [ 8, 11, 15],
                     [11,  3, 10],
                     [19,  4, 12],
                     [11,  8,  5],
                     [19,  5, 15]]
        Four shot tuning-20181115:
                     [array([15,  7,  3,  8]),
                     array([ 7, 17,  6, 14]),
                     array([10, 17,  7,  4]),
                     array([ 9,  5, 14,  3]),
                     array([ 6,  2, 19,  1]),
                     array([15, 12,  1,  3])] -> tuning acc: 83.33%
        Five shot tuning-20181115: 90%
                     [array([18, 17,  7,  5, 12]),
                     array([ 1,  7, 15, 17,  8]),
                     array([ 5,  1, 17, 19,  9]),
                     array([ 5,  4,  3, 15,  0]),
                     array([ 2, 16,  8,  1,  3]),
                     array([13, 17,  6, 15,  4])] -> 90.00%
         seven shot tuning-20181115:
         Best record:[[9, 12, 2, 13, 17, 16, 6],
                     [5, 8, 17, 11, 18, 10, 4],
                     [13, 8, 10, 4, 9, 5, 14],
                     [18, 14, 11, 12, 5, 3, 2],
                     [11, 9, 10, 15, 0, 2, 17],
                     [2, 18, 10, 15, 1, 16, 9]]
        '''
        # path = f"./Sample_index/sample_index_record_for_{self.nshots}_shots_711_20181115.mat"
        path = self.matPath
        self.record = loadmat(path)['record']
        self.tuningAcc = loadmat(path)['val_acc']
    # def getFineTunedModelPath( self ):
    #     if self.train_dir == 'E:/Cross_dataset/20181115':
    #         if self.domain_selection == (6,1,1):
    #             acc = [51.8,69.3,83.4,85.2,90.0,87.7,87.7]
    #             self.tunedModel_path = f'./models/widar_fineTuned_model_20181115_{self.nshots}shots_{acc[self.nshots-1]}%.h5'
    #         elif self.domain_selection == (7,1,1):
    #             acc = []
    #             self.tunedModel_path = f'./models/widar_fineTuned_model_20181115_{config.nshots}shots_711_{acc[self.nshots-1]}%.h5'
    def initGPU(self):
        gpus = tf.config.experimental.list_physical_devices( 'GPU' )
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth( gpu, True )
            except RuntimeError as e:
                print( e )