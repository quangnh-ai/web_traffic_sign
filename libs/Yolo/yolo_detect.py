import cv2
import numpy as np
import os
import random
import imutils
import time
import glob

net = cv2.dnn.readNet("yolov4_Traffic_last.weights", "yolov4_Traffic.cfg")

# Name custom object
classesFile = "obj.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def detect_image(img):
    
    images=img.split('/')[-1]
    #print(images)
    output_path = os.path.join("results", images)
    img=cv2.imread(img)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
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
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-2), font, 1, color, 2)
            box_new=[x,x+w,y,y+h]#bbox on pascalVOC
            new_bbox=convert((width,height),box_new)#bbox on yolo
            print(class_ids[i],new_bbox[0],new_bbox[1],new_bbox[2],new_bbox[3])
            name=images.split(".")[0]
            name="results/"+name+".txt"
            with open(name,'a',encoding = 'utf-8') as f:
                f.write(str(class_ids[i])+" ")   
                f.write(str(new_bbox[0])+" ")
                f.write(str(new_bbox[1])+" ")
                f.write(str(new_bbox[2])+" ")
                f.write(str(new_bbox[3]))
                f.write("\n") 

    return img, output_path

def detect_video(video_path):
    # start detect video
    cap = cv2.VideoCapture(video_path)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = cap.read()
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter("results/out1.avi",codec,15,(WIDTH,HEIGHT))
    cap.release()
    counts = 0
    cap = cv2.VideoCapture(video_path)
    while (True):
        ret, frame = cap.read() 

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
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
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-2), font, 1, color, 2)
    
        counts += 1
        #cv2.imshow('detection', frame)
        writer.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.exists("results"):
        os.mkdir("results")

    # Detect video
    #detect_video('sample_01.mp4')

    # Detect image
    image, image_path = detect_image("4675.png")
    cv2.imshow("demo", image)
    cv2.waitKey(0)
    # for dir_path in glob.glob('images/*.jpg'):
    #     # Detec image
    #     image, image_path = detect_image(dir_path)
    #     cv2.waitKey(0)