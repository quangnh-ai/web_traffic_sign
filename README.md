# CS338-Pattern Recognition Project

## Windows:
### Create virtual Venv: 
python -m venv venv

venv\Scripts\activate

### install requirments:
pip install -r requirements.txt

### install detectron2:
- Step 1: cd libs\detectron2
- Step 2: python setup.py build develop

### Download model weights:
- Retinanet: https://drive.google.com/file/d/15vaA8QHE6g2vL_6x2fv3dX2PPjAoBMOj/view?usp=sharing
- Faster RCNN: https://drive.google.com/file/d/16iDWG_BYBAaITTg8RkXdqiQzABm_5tBf/view?usp=sharing
- Yolov4:
  + model config: https://drive.google.com/file/d/1pQl2-TrLfmmZRokaBjlrt_pgHYRiyZym/view
  + model weights: https://drive.google.com/file/d/1Z4va8nPor3E3RCFi30EVwJh2PsPIpQIN/view?usp=sharing
  + obj.names: https://drive.google.com/file/d/16Ug2B-QWAmmnGkP6pPvUV1x_WrLSWnsa/view?usp=sharing

### Run App:
- Step 1: Create a folder name models
- Step 2: Move retinanet and faster rcnn weights in to models folder
- Step 3: Create a sub folder names yolo
- Step 4: move model config, weights, obj.names files to yolo folder 
- Step 5: Run cmd line: uvicorn app:app
