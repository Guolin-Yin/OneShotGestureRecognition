import numpy as np
class getConfig:
    def __init__(self):
        self.train_dir = './20181116'
        self.eval_dir = './20181115'
        self.lr = 1e-3
    def get_params(self):
        return {train_dir: self.train_dir,
                eval_dir: self.eval_dir}