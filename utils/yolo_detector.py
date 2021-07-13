import cv2
import numpy as np
import os
import random

class Yolo_Detector:
    
    def __init__(self, model_cfg_path, model_weight_path, obj_names_path):
        self.model = cv2.dnn.readNet(model_weight_path, model_cfg_path)

        self.classes = None
        with open(obj_names_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def predict(self, image):
        
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        total = 0
        class_count = [0] * 7

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, label, (x, y-2), font, 1, color, 2)

                total += 1
                class_count[class_ids[i]] += 1
        
        cv2.imwrite('static/images/output.jpg', image)
        return total, class_count