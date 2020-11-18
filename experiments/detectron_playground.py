import torch, torchvision
import detectron2
import os
import cv2
import numpy as np
from models.OdDetectronWrapper import OdDetectronWrapper
from models.SegmentationDetectronWrapper import SegmentationDetectronWrapper
import models.faiss as faiss
from utils.image_preprocess import img_2_bb_img


is_extract_bb_test = False
is_segment_ring = False
is_e2e_test = True

if is_e2e_test:
    query_img = 'data/query_images/rings/plain_gold.jpeg'
    img = cv2.imread(query_img)

    model_path = 'resources/object_detection/ring_11_11/model_final.pth'
    metadata_path = 'resources/object_detection/ring_11_11/metadata.pkl'
    detection_model = OdDetectronWrapper(model_path, metadata_path)

    model_path = 'resources/segmentation/ring_16_11/model_final.pth'
    metadata_path = 'resources/segmentation/ring_16_11/metadata.pkl'
    seg_model = SegmentationDetectronWrapper(model_path, metadata_path)

    bbs = detection_model.detect(img)
    for bb in bbs:
        bb_img = img_2_bb_img(img, bb)
        polygon = seg_model.segment_img(bb_img)
        faiss.query_index(bb_img)



if is_segment_ring:
    model_path = 'resources/segmentation/ring_16_11/model_final.pth'
    metadata_path = 'resources/segmentation/ring_16_11/metadata.pkl'
    model_wrapper = SegmentationDetectronWrapper(model_path, metadata_path)

    dir_path = 'data/our_jewellery/ring/train/'
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('g'):
            continue
        segmentation_img = model_wrapper.segment_img(dir_path + file_name)


if is_extract_bb_test:
    model_path = 'resources/object_detection/ring_11_11/model_final.pth'
    metadata_path = 'resources/object_detection/ring_11_11/metadata.pkl'
    model_wrapper = OdDetectronWrapper(model_path, metadata_path)

    dir_path = 'data/our_jewellery/ring/train/'
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('g'):
            continue
        bb_imgs = model_wrapper.detect(dir_path + file_name)

    img_path = 'data/our_jewellery/ring/dev/r12.jpeg'
    bb_imgs = model_wrapper.detect(img_path)
