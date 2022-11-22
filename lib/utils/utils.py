import os
import os.path as osp

def set_log(args, cfg):
    dataset_name = cfg.DATASET.NAME
    cfg_name = args.cfg.split(os.sep)[-1].replace('.yaml','')
    output_dir = osp.join(cfg.OUTPUT_DIR, dataset_name, cfg_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
