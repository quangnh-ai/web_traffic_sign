from utils.detector import Detector

import cv2

detector = Detector("libs/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml", 'models/model_final_faster_rcnn.pth')
image = cv2.imread('417.jpg')
detector.predict(image)