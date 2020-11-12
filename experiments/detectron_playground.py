import torch, torchvision
import detectron2
import os
import numpy as np
import os
import json
import cv2
import random
import wget

from detectron2.utils.logger import setup_logger
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

from models.DetectronWrapper import DetectronWrapper


is_extract_bb_test = True

if is_extract_bb_test:
    model_path = 'resources/od_predictors/ring_11_11/model_final.pth'
    metadata_path = 'resources/od_predictors/ring_11_11/metadata.pkl'
    model_wrapper = DetectronWrapper(model_path, metadata_path)

    dir_path = 'data/our_jewellery/ring/train/'
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('g'):
            continue
        bb_imgs = model_wrapper.extract_bb_imgs(dir_path + file_name)

    img_path = 'data/our_jewellery/ring/dev/r12.jpeg'
    bb_imgs = model_wrapper.extract_bb_imgs(img_path)
