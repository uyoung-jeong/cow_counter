import numpy as np

class Evaluator():
    def __init__(self,cfg):
        self.metrics = cfg.TEST.METRICS

    def evaluate(self, pred, gt):
        score_dict = dict()

        # rmse
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        score_dict['rmse'] = rmse

        # ap, ar
        pred_int = np.around(pred).astype(int) # discretize

        cat_data = np.stack((pred_int,gt))
        tp = np.min(cat_data, axis=0)
        fp = np.clip(pred_int - gt, a_min=0, a_max=None)
        fn = np.clip(gt - pred_int, a_min=0, a_max=None)

        ap = np.mean(tp/(tp+fp))
        ar = np.mean(tp/(tp+fn))
        score_dict['ap'] = ap
        score_dict['ar'] = ar

        return score_dict
