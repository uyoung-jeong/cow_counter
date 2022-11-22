from sklearn.svm import SVR as skSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SVR():
    def __init__(self, cfg):
        self.kernel = cfg.MODEL.SVR.KERNEL
        self.gamma = cfg.MODEL.SVR.GAMMA
        self.coef0 = cfg.MODEL.SVR.COEF0
        self.tol = cfg.MODEL.SVR.TOL
        self.c = cfg.MODEL.SVR.C
        self.eps = cfg.MODEL.SVR.EPS
        self.cache_size = cfg.MODEL.SVR.CACHE_SIZE
        self.max_iter = cfg.TRAIN.MAX_ITER

        self.model = make_pipeline(StandardScaler(),
                    skSVR(kernel=self.kernel,
                        gamma=self.gamma,
                        coef0=self.coef0,
                        tol=self.tol,
                        C=self.c,
                        epsilon=self.eps,
                        cache_size=self.cache_size,
                        max_iter=self.max_iter))

    def fit(self,x,y):
        return self.model.fit(x,y)

    def predict(self,x):
        return self.model.predict(x)
