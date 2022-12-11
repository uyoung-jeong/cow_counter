from dataset import Dataset

from .svr import SVR
from .mlp import MLP
from .mlp_svr import MLP_SVR
from .evaluator import Evaluator


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        # model
        self.model_class = cfg.MODEL.CLASS

        # dataset
        self.dataset_name = cfg.DATASET.NAME
        self.num_fold = 5
        self.dataset = Dataset(cfg)
        self.evaluator = Evaluator(cfg)

    def k_fold_evaluation(self):
        mean_fold_scores = dict()

        for ki in range(self.num_fold):
            train_data, test_data = self.dataset.split_5fold(ki)
            print(f"{len(train_data['img_ids'])} samples for train, {len(test_data['img_ids'])} samples for test")

            train_x, train_y = train_data['feats'], train_data['num_target']
            test_x, test_y = test_data['feats'], test_data['num_target']

            # initialize new model
            self.model = eval(self.model_class)(self.cfg)

            # train model
            self.model = self.model.fit(train_x,train_y)

            # evaluate
            preds = self.model.predict(test_x)

            # denormalize prediction
            preds = self.dataset.denormalize(x=None,y=preds)

            score_dict = self.evaluator.evaluate(preds, test_y)

            print(f"{ki}th fold. rmse:{score_dict['rmse']:.4f}, ap:{score_dict['ap']:.4f}, ar:{score_dict['ar']:.4f}")

            for k,v in score_dict.items():
                if k not in mean_fold_scores.keys():
                    mean_fold_scores[k] = []
                mean_fold_scores[k] += [v]

        for k,v in mean_fold_scores.items():
            mean_fold_scores[k] = sum(v)/len(v)
        print("5-fold average metrics")
        print(f"rmse:{mean_fold_scores['rmse']:.4f}, ap:{mean_fold_scores['ap']:.4f}, ar:{mean_fold_scores['ar']:.4f}")

        print('='*100)
