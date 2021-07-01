from fastapi import responses
from utils.detector import Detector

import cv2
import os
import shutil

from fastapi import FastAPI, Request, File, UploadFile
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
async def recognition(request: Request, image: UploadFile = File(...)):
    with open("./static/images/destination.jpg", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    image = cv2.imread('static/images/destination.jpg')
    
    detector = Detector('libs/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml', 'models/weights/model_final_retinanet.pth')
    detector.predict(image)
    
    return templates.TemplateResponse("result.html",{
        "request": request
    })