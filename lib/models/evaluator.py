import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class Evaluator():
    def __init__(self,cfg):
        self.metrics = cfg.TEST.METRICS

    def evaluate(self, pred, gt):
        num_sample = len(gt)
        score_dict = dict()

        # rmse
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        score_dict['rmse'] = rmse

        # ap, ar
        pred_int = np.around(pred).astype(int) # discretize

        cat_data = np.stack((pred_int,gt))
        tp = np.clip(np.min(cat_data, axis=0), a_min=0, a_max=None)
        fp = np.clip(pred_int - gt, a_min=0, a_max=None)
        fn = np.clip(gt - pred_int, a_min=0, a_max=None)

        #ap = np.mean(tp/(tp+fp)) # sometimes tp+fp=0
        tp_fp = tp+fp
        tp_fp_valid = tp_fp>0
        ap_sum = np.sum(tp[tp_fp_valid]/tp_fp[tp_fp_valid])
        ap = ap_sum/num_sample

        ar = np.mean(tp/(tp+fn))
        score_dict['ap'] = ap
        score_dict['ar'] = ar

        return score_dict
    
   