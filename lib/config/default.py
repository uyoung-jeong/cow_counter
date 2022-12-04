import os
from yacs.config import CfgNode as CN

_C = CN()
_C.CFG_NAME = ''
_C.OUTPUT_DIR = 'output'
_C.WORKERS = 4
_C.PRINT_FREQ = 20
#_C.AUTO_RESUME = False
#_C.PIN_MEMORY = True

"""
# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = True
_C.CUDNN.ENABLED = True
"""

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.CLASS = 'SVR' # ['SVR', 'MLP', 'MLP_SVR']
_C.MODEL.SVR = CN()
_C.MODEL.SVR.KERNEL = 'rbf'
_C.MODEL.SVR.GAMMA = 'scale'
_C.MODEL.SVR.COEF0 = .0
_C.MODEL.SVR.TOL = 1.0E-6
_C.MODEL.SVR.C  = 1.0
_C.MODEL.SVR.EPS  = 0.1
_C.MODEL.SVR.CACHE_SIZE = 200

_C.MODEL.MLP = CN()
_C.MODEL.MLP.HIDDEN_LAYER_SIZES = [100]
_C.MODEL.MLP.ACTIVATION = 'relu'
_C.MODEL.MLP.SOLVER = 'adam'
_C.MODEL.MLP.ALPHA = 0.0001
_C.MODEL.MLP.BATCH_SIZE = 'auto'
_C.MODEL.MLP.LEARNING_RATE = 'constant'
_C.MODEL.MLP.LEARNING_RATE_INIT = 0.001
_C.MODEL.MLP.POWER_T = 0.5
# _C.MODEL.MLP.MAX_ITER = 300
_C.MODEL.MLP.SHUFFLE = True
_C.MODEL.MLP.TOL = 1e-3
_C.MODEL.MLP.VERBOSE=True
_C.MODEL.MLP.EARLY_STOPPING = False
_C.MODEL.MLP.BETA1 = 0.9
_C.MODEL.MLP.BETA2 = 0.999

_C.LOSS = CN()

# DATASET related params
_C.DATASET = CN()
_C.DATASET.NAME = 'coco_cow_nfeatures_128'
_C.DATASET.ROOT = 'data'
_C.DATASET.IMAGE_SIZE = 128
_C.DATASET.TRAIN = CN()
_C.DATASET.VAL = CN()
_C.DATASET.TEST = CN()
_C.DATASET.MAX_INST = 14

# train
_C.TRAIN = CN()
_C.TRAIN.LR_INIT = 0.01
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 5.0e-4
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.START_ITER = 0
_C.TRAIN.MAX_ITER = 300

# testing
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''
_C.TEST.METRICS = ['rmse', 'ap', 'ar']

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
