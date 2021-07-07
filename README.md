# CS338-Pattern Recognition Project

## Windows:
### Create virtual Venv: 
python -m venv venv

venv\Scripts\activate

### install requirments:
pip install -r requirements.txt

### install detectron2:
cd libs\detectron2

python setup.py build develop

### Download model weights:
Retinanet: https://drive.google.com/file/d/15vaA8QHE6g2vL_6x2fv3dX2PPjAoBMOj/view?usp=sharing

Faster RCNN: https://drive.google.com/file/d/16iDWG_BYBAaITTg8RkXdqiQzABm_5tBf/view?usp=sharing

### Run App:
- Step 1: Create a folder name models and move all model weights to this folder
- Step 2: Run cmd line: uvicorn app:app
