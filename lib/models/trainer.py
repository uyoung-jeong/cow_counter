from dataset import Dataset
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

            #np.clip(preds, a_min=0, a_max=14)

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
        
        fig = px.scatter(x=test_y, y=preds, labels={'x': 'ground truth', 'y': 'predictions'})
        np.clip(preds, a_min=0, a_max=14)
        fig.add_shape(
                type="line", line=dict(dash='dash'),
                x0=test_y.min(), y0=test_y.min(),
                x1=test_y.max(), y1=test_y.max(),
            )
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )
        fig.update_xaxes(
            range=[0,15],  # sets the range of xaxis
            constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
            )
        fig.update_layout(autosize=False,
        width=800, height=800,
        title={
        'text': "MLP+SVR (HL=100,poly) Performance",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        font=dict(
        size=15)
        )

        fig.show()

        



    