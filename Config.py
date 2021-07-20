import numpy as np
class getConfig:
    def __init__(self):
        self.train_dir = [
                            'E:/Widar_dataset_matfiles/20181109/User1',
                            'E:/Widar_dataset_matfiles/20181112/User1',
                            'E:/Widar_dataset_matfiles/20181109/User2',
                            'E:/Widar_dataset_matfiles/20181112/User2',
                            # 'E:/Widar_dataset_matfiles/20181115'
                            ]
        self.eval_dir = 'E:/Widar_dataset_matfiles/20181121/test_user1_r3'
        self.lr = 1e-3
        self.num_classes = 16
    def get_params(self):
        return {train_dir: self.train_dir,
                eval_dir: self.eval_dir}