import inspect
import numpy as np
import os
import glob
from sklearn.neural_network import MLPRegressor


class MLP():
    def __init__(self, cfg):
        self.cfg = cfg
        self.hidden_layer_sizes = cfg.MODEL.MLP.HIDDEN_LAYER_SIZES
        self.activation = cfg.MODEL.MLP.ACTIVATION
        self.solver = cfg.MODEL.MLP.SOLVER
        self.alpha = cfg.MODEL.MLP.ALPHA
        self.batch_size = cfg.MODEL.MLP.BATCH_SIZE
        self.learning_rate = cfg.MODEL.MLP.LEARNING_RATE
        self.learning_rate_init = cfg.MODEL.MLP.LEARNING_RATE_INIT
        self.power_t = cfg.MODEL.MLP.POWER_T
        self.max_iter = cfg.TRAIN.MAX_ITER
        self.shuffle = cfg.MODEL.MLP.SHUFFLE
        self.tol = cfg.MODEL.MLP.TOL
        self.verbose = cfg.MODEL.MLP.VERBOSE
        self.early_stopping = cfg.MODEL.MLP.EARLY_STOPPING
        self.beta1 = cfg.MODEL.MLP.BETA1
        self.beta2 = cfg.MODEL.MLP.BETA2


        self_attr = list(self.__dict__.keys())
        func_attr = list(inspect.signature(MLPRegressor).parameters)
        alter_dict = dict()

        for attr in self_attr:
            if attr in func_attr:
                alter_dict[attr] = getattr(self, attr)

        self.model = MLPRegressor(**alter_dict)

    def fit(self,x,y):
        result = self.model.fit(x,y)
        return result

    def predict(self,x):
        return self.model.predict(x)
