import torch
import os
import pickle
import cv2
from detectron2.utils.logger import setup_logger
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from utils.image_preprocess import img_2_bb_img



class OdDetectronWrapper:

    def __init__(self, model_path, metadata_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
        try:
            self.model = DefaultPredictor(cfg)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        except Exception as e:
            print('error loading model and metadata')
            print(e)

    def detect(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)

        detections = self.model(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        bbs = detections['instances']._fields['pred_boxes'].tensor.detach().cpu().numpy()
        return bbs





