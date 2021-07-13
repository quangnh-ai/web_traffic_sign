from fastapi import responses
from utils.detector import Detector
from utils.yolo_detector import Yolo_Detector

import cv2
import os
import shutil

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

#index page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })

@app.post("/result", response_class=HTMLResponse)
async def recognition(request: Request, image: UploadFile = File(...), model: str = Form(...)):
    with open("./static/images/destination.jpg", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if model=='faster_rcnn':
        detector = Detector('libs/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', 'models/model_final_faster_rcnn.pth')
    elif model == 'retinanet':
        detector = Detector('libs/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml', 'models/model_final_retinanet.pth')
    elif model == 'yolo':
        detector = Yolo_Detector('models/yolo/yolov4_Traffic.cfg', 'models/yolo/yolov4_Traffic_last.weights', 'models/yolo/obj.names')

    image = cv2.imread('static/images/destination.jpg')
    
    total, class_count = detector.predict(image)

    result = {
        "Tổng số biển báo phát hiện được": total,
        "Cấm ngược chiều": class_count[0],
        "Cấm dừng và đỗ": class_count[1],
        "Cấm rẽ": class_count[2],
        "Giới hạn tốc độ": class_count[3],
        "Cấm còn lại": class_count[4],
        "Nguy hiểm": class_count[5],
        "Hiệu lệnh": class_count[6]
    }
    
    return templates.TemplateResponse("result.html",{
        "request": request,
        "result": result
    })