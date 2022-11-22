import random
import numpy as np
import argparse
from yacs.config import CfgNode

import os
import os.path as osp
import sys
base_dir = osp.dirname(__file__)
lib_path = osp.join(base_dir, 'lib')
sys.path.insert(0, base_dir)
sys.path.insert(0, lib_path)

from config.default import _C
from models import Trainer

from utils import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/svr.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('opts',
                        help="commandline cfg options",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = _C.clone()
    # update cfg
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return args, cfg

def main():
    args, cfg = parse_args()
    output_dir = utils.set_log(args, cfg)
    tb_dir = osp.join(output_dir, 'tb')

    # seed initialization
    random.seed(args.seed)
    np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    trainer = Trainer(cfg)
    trainer.k_fold_evaluation()

if __name__ == '__main__':
    main()
