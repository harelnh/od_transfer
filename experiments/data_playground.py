import cv2
import os
import json
from utils.image_preprocess import img_2_bb_img, get_img_bbs_coco


def create_bb_dataset(in_dir_path, out_dir_path):

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    with open(in_dir_path + '/instances_default.json') as f:
        annotations_obj = json.load(f)
    for file_name in os.listdir(in_dir_path):
        img = cv2.imread(in_dir_path + '/' + file_name)
        bb_arr = get_img_bbs_coco(file_name, annotations_obj)
        for idx, bb in enumerate(bb_arr):
            bb_img = img_2_bb_img(img, bb)
            # cv2.imshow('bla',bb_img)
            # cv2.waitKey()
            cv2.imwrite(out_dir_path + '/' + file_name.split('.')[0] + '_' + str(idx) + '.jpg', bb_img)


is_create_bb_dataset = True

if is_create_bb_dataset:
    in_dir_path = 'data/our_jewellery/ring/train'
    out_dir_path = 'data/our_jewellery/ring/train_bb'
    create_bb_dataset(in_dir_path, out_dir_path)