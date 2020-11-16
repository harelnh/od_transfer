import cv2
import math


def img_2_bb_img(img, bb):
    x = math.floor(bb[0])
    y = math.floor(bb[1])
    w = math.ceil(bb[2])
    h = math.ceil(bb[3])
    bb = img[y:y+h,x:x+w,:]
    return bb


def get_img_bbs_coco(file_name, coco_instances_obj):
    bbs = []
    try:
        img_id = next((x for x in coco_instances_obj['images'] if x['file_name'] == file_name), None)['id']
        img_annotations = [x for x in coco_instances_obj['annotations'] if x['image_id'] == img_id]
        bbs = [ann['bbox'] for ann in img_annotations]
    except:
        pass

    return bbs
