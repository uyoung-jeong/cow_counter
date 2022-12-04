import os
import os.path as osp
import pickle
import numpy as np

class Dataset():
    def __init__(self,cfg):
        dataset_path = osp.join(cfg.DATASET.ROOT, f"{cfg.DATASET.NAME}.pkl")
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        self.data_keys = list(self.data.keys())
        self.max_inst = cfg.DATASET.MAX_INST

    def __len__(self):
        return len(self.data['img_ids'])

    def denormalize(self,x=None,y=None):
        if y is not None:
            return y * self.max_inst

    def split_fold(self, start_idx, end_idx):
        fold_data = dict()
        for key in self.data_keys:
            fold_data[key] = self.data[key][start_idx:end_idx]
        return fold_data

    def concat_dict(self, dicts):
        cat_dict = dict()
        keys = list(dicts[0].keys())
        for k in keys:
            cat_dict[k]=[]
        for di, dic in enumerate(dicts):
            for k in keys:
                if isinstance(dic[k],np.ndarray):
                    if len(cat_dict[k])==0:
                        cat_dict[k] = dic[k]
                    else:
                        cat_dict[k] = np.concatenate((cat_dict[k], dic[k]),axis=0)
                else:
                    cat_dict[k] += dic[k]
        return cat_dict

    # ki: fold idx
    def split_5fold(self, ki, skip_val=True):
        n_train_assigned = 0

        data_len = len(self)
        split_idxs = [int(data_len*(ii/5)) for ii in range(5)] + [data_len]

        train_chunk = []
        test_chunk = []
        if not skip_val:
            n_train_chunk = 3
            val_chunk = []
            for i in range(5):
                start_idx = split_idxs[i]
                end_idx = split_idxs[i+1]
                if (i == ki) and not skip_val:
                    train_chunk.append(self.split_fold(start_idx, end_idx))
                if n_train_assigned<n_train_chunk:
                    val_chunk.append(self.split_fold(start_idx, end_idx))
                    n_train_assigned += 1
                else:
                    test_chunk.append(self.split_fold(start_idx, end_idx))

            # concat
            train_data = self.concat_dict(train_chunk)
            test_data = self.concat_dict(test_chunk)
            val_data = self.concat_dict(val_chunk)

            train_data['feats'] = np.stack(train_data['feats'])
            train_data['num_target'] = np.clip(np.stack(train_data['num_target']),a_min=0, a_max=self.max_inst)/self.max_inst
            val_data['feats'] = np.stack(val_data['feats'])
            val_data['num_target'] = np.clip(np.stack(val_data['num_target']),a_min=0, a_max=self.max_inst)
            test_data['feats'] = np.stack(test_data['feats'])
            test_data['num_target'] = np.clip(np.stack(test_data['num_target']),a_min=0, a_max=self.max_inst)
            return train_data, val_data, test_data
        else: # train 4, test 1
            n_train_chunk = 4
            for i in range(5):
                start_idx = split_idxs[i]
                end_idx = split_idxs[i+1]
                if (i == ki):
                    test_chunk.append(self.split_fold(start_idx, end_idx))
                else:
                    train_chunk.append(self.split_fold(start_idx, end_idx))
                    n_train_assigned += 1

            # concat
            train_data = self.concat_dict(train_chunk)
            test_data = self.concat_dict(test_chunk)

            train_data['feats'] = np.stack(train_data['feats'])
            train_data['num_target'] = np.clip(np.stack(train_data['num_target']),a_min=0, a_max=self.max_inst)/self.max_inst
            test_data['feats'] = np.stack(test_data['feats'])
            test_data['num_target'] = np.clip(np.stack(test_data['num_target']),a_min=0, a_max=self.max_inst)

            return train_data, test_data
