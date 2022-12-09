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
        self.mlp.fit(x,y)
        mlp_preds = self.mlp.predict(x).reshape(-1,1)
        self.svr.fit(mlp_preds, y)
        return self

    def predict(self, x):
        mlp_preds = self.mlp.predict(x).reshape(-1,1)
        return self.svr.predict(mlp_preds)
