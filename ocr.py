import cv2
import numpy as np
import imutils
import easyocr
import pandas as pd
from utils import *
import base64
def inference(img):
   #rotate image if needed 
    image=img.copy()
    image = imutils.resize(image, width=800)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    reader = easyocr.Reader(['es','en'])
    thresholded = cleanImage(image)
    bounds = reader.readtext(thresholded)
    result=draw_boxes(image, bounds)
    face=grabFace(image)
    avartar=[]
    if face is not None:
        _, imgBuffer = cv2.imencode(".jpg", imutils.resize(face, 240))
        avartar= base64.b64encode(imgBuffer).decode("utf-8")
    
    return [result, pd.DataFrame(bounds), avartar]

