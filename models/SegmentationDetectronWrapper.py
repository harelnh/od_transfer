import torch
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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



class SegmentationDetectronWrapper:

    def __init__(self, model_path, metadata_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
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


    def segment_img(self, im, is_plot = False):

        outputs = self.model(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=self.metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))  # Passing the predictions to CPU from the GPU

        if is_plot:
            plt.imshow(v.get_image()[:, :, ::-1])
            plt.show()

        return outputs

    def segment_img_file(self, im_path):
        im = cv2.imread(im_path)
        self.segment_img(im)




