import inspect
import numpy as np
from sklearn.preprocessing import StandardScaler
from .svr import SVR
from .mlp import MLP


class MLP_SVR():
    def __init__(self, cfg):
        self.mlp = MLP(cfg)
        self.svr = SVR(cfg)

    def fit(self, x, y):
        self.mlp.model.max_iter = 2
        self.mlp.fit(x,y, save=False)
        self.load_weight()
        
        # getting mlp's logit
        train_logit = self.get_logit(x)
        train_logit_target = y

        self.svr = self.svr.fit(train_logit, train_logit_target)
        return self
    
    def predict(self, x):
        test_logit = self.get_logit(x)
        scaler = StandardScaler()
        scaler.fit(test_logit)
        scaled_logit = scaler.transform(test_logit)
        # print(tt1_2[:3])
        # print(test_logit[:3])
        return self.svr.predict(scaled_logit)

    def get_logit(self, x):
        logit = self.mlp.predict(x).reshape(-1,1)
        return logit

    def load_weight(self):
        self.mlp.load_weight(self.idx)
