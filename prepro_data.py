import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
import json
import shutil
from tqdm import tqdm
import cv2
import pickle

id_dict = {'horse':19,
            'sheep':20, # 9609 anns, 1529 images
            'cow':21} # 8147 anns, 1968 images
# {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}

def prepro_data(base_dir, name='cow'):
    anno_path = osp.join(base_dir, 'annotations', 'instances_train2017.json')

    cat_id = id_dict[name]

    with open(anno_path, 'r') as f:
        data = json.load(f) # ['info', 'licenses', 'images', 'annotations', 'categories']
    coco = COCO(anno_path)

    # trim category part
    data['categories'] = [e for e in data['categories'] if e['name']==name]

    # trim annotations
    sub_anns = [e for e in data['annotations'] if e['category_id']==cat_id]
    #data['annotaions'] = sub_anns

    # trim images
    target_image_ids = np.array([e['image_id'] for e in sub_anns])
    target_image_ids = np.unique(target_image_ids)

    img_anns = [e for e in data['images'] if e['id'] in target_image_ids]
    #data['images'] = img_anns
    """
    # visualize
    vis_dir = 'vis'
    os.makedirs(vis_dir, exist_ok=True)
    for ei,e in enumerate(tqdm(img_anns, desc='visualization')):
        if ei % 2 == 0:
            continue
        anns_per_img = coco.loadAnns(coco.getAnnIds([e['id']]))
        cat_anns_per_img = [e for e in anns_per_img if e['category_id']==cat_id]
        num_cat_anns = len(cat_anns_per_img)
        filename = e['file_name']
        src_path = osp.join(base_dir, 'train2017', filename)
        dst_path = osp.join(vis_dir, filename.replace('.jpg',f'_{num_cat_anns}.jpg'))
        shutil.copyfile(src_path, dst_path)
    """

    # feature extraction
    nfeatures = 128
    filter_thres = 4000 # if descriptor size is smaller than threshold, exclude
    feature_size = 128*(32+2) # may need to modify number if another feature is added
    orb = cv2.ORB_create(
                nfeatures=nfeatures,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20,
                )

    img_ids = []
    #imgs = []
    feats = []
    img_fnames = []
    ann_ids = []
    bboxes = []
    num_targets = []

    for img_ann in tqdm(img_anns):
        img_fname = img_ann['file_name']
        imgid = img_ann['id']

        # read image
        src_img_path = osp.join(base_dir, 'train2017', img_fname)
        img = cv2.imread(src_img_path)
        H,W,C=img.shape

        # find the keypoints with ORB
        kpts, descriptions = orb.detectAndCompute(img, None)

        feat = []
        for kp, desc in zip(kpts, descriptions):
            pt = kp.pt # [width axis, height axis]
            desc = desc.reshape(-1)

            # normalize
            pt = np.array([pt[0]/W, pt[1]/H])
            desc = desc/255

            feat.append(np.concatenate((pt, desc)))
        feat = np.hstack(feat)
        cur_feat_size = feat.size
        if cur_feat_size<filter_thres:
            continue

        if cur_feat_size < feature_size:
            feat = np.hstack((feat, np.zeros(feature_size-cur_feat_size)))

        feats.append(feat) # [(pt (2), desc (32)) x 128]

        # load ann
        anns_per_img = coco.loadAnns(coco.getAnnIds([imgid]))
        cat_anns_per_img = [e for e in anns_per_img if e['category_id']==cat_id]
        num_target = len(cat_anns_per_img)
        cat_ann_ids = [e['id'] for e in cat_anns_per_img]
        cat_bboxes = [e['bbox'] for e in cat_anns_per_img]

        # append to list
        img_ids.append(imgid)
        #imgs.append(img)
        img_fnames.append(img_fname)
        ann_ids.append(cat_ann_ids)
        bboxes.append(cat_bboxes)
        num_targets.append(num_target)

    # save sub dataset file
    out_dict = {'img_ids': img_ids,
                #'imgs': imgs,
                'feats': feats,
                'img_fnames': img_fnames,
                'ann_ids': ann_ids,
                'bboxes': bboxes,
                'num_target': num_targets}
    out_path = osp.join("data", f"coco_{name}_nfeatures_{nfeatures}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(out_dict, f)

    print(f"pickle file written at: {out_path}")

if __name__ == "__main__":
    base_dir = osp.join('/syn_mnt/uyoung/human/coco')
    prepro_data(base_dir=base_dir, name='cow')
