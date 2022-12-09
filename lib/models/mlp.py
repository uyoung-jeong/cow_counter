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
        if save:
            self.save_weight()
        return result
    
    def predict(self,x):
        return self.model.predict(x)
    
    def save_weight(self):
        if os.path.isdir(f'lib/weights') and self.cfg.MODEL.MLP.SAVE_NAME:
            split_name = self.cfg.MODEL.MLP.SAVE_NAME.split('.')[0]
            check_name = split_name + '_5.npy'

            if os.path.isfile(f'lib/weights/{check_name}'):
                del_old_files = glob.glob(f'lib/weights/{split_name}*', recursive=True)
                print('del_old_files : ', del_old_files)
                for old_file in del_old_files:
                    try:
                        os.remove(old_file)
                    except:
                        pass

            prior_list = sorted(glob.glob1('lib/weights/', f'{split_name}*'))
            # print('prior_list :   ', prior_list)
            if len(prior_list) == 0:
                save_idx = 1
            else:
                save_idx = int(prior_list[-1].split('_')[2][0])+1
            save_name = split_name+f'_{save_idx}.npy'

            save_dict = {'coefs':self.model.coefs_, 'intercepts':self.model.intercepts_}
            with open(f'lib/weights/{save_name}', 'wb') as f:
                np.save(f, save_dict, allow_pickle=True)
                print(save_name, 'saved')

        else:
            print(self.cfg.MODEL.MLP.SAVE_NAME)

    def load_weight(self, idx):
        file_name = self.cfg.MODEL.MLP.SAVE_NAME.split('.')[0]+f"_{idx+1}.npy"
        with open(f'lib/weights/{file_name}', 'rb') as f:
            weights = np.load(f, allow_pickle=True)
        weights = weights.item()
        num_coefs = len(weights['coefs'])

        for idx in range(num_coefs):
            self.model.coefs_[idx] = weights['coefs'][idx]
            self.model.intercepts_[idx] = weights['intercepts'][idx]
    
