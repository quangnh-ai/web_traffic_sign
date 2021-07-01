import os
import cv2
import json
import random
import itertools
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Detector:

    def __init__(self, model_cfg_path, model_weight_path):

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_cfg_path)
        self.cfg.MODEL.WEIGHTS = model_weight_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
        self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, image):
        output = self.predictor(image)
        boxes = output['instances'].pred_boxes
        scores = output['instances'].scores
        class_ids = output['instances'].pred_classes

        classes = ['Cam_nguoc_chieu',
                    'Cam_dung_va_do',
                    'Cam_re',
                    'Gioi_han_toc_do',
                    'Cam_con_lai',
                    'Nguy_hiem',
                    'Hieu_lenh']
        
        for i in range(len(class_ids)):
            if scores[i] >= 0.5:
                for box in boxes[i]:
                    start = (int(box[0]), int(box[1]))
                    end = (int(box[2]), int(box[3]))
                    color = int(class_ids[i])

                    cv2.rectangle(image, start, end, (random.randint(0, 255), random.randint(0, 255), 255), 1)
                    cv2.putText(image, str(classes[color]), start, cv2.FONT_HERSHEY_PLAIN, 1, (random.randint(0, 255), random.randint(0, 255), 255), 2)

        cv2.imwrite('static/images/output.jpg', image)