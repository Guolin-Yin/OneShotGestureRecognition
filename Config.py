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
        self.N_train_classes = 250
        self.batch_size = 32
        self.input_shape = [200,60,3]
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