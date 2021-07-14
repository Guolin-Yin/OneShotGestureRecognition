import numpy as np
class getConfig:
    def __init__(self):
        self.train_dir = './20181116'
        self.eval_dir = './20181115'
    def get_params(self):
        return {tran_dir: self.train_dir,
                eval_dir: self.eval_dir}