import cv2 
import numpy as np
import fastapi
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
class Analyzer(BaseModel):
    filename: str
    img_dimensions: str
    encoded_img: str

app = FastAPI()

@app.post("/analyze", response_model=Analyzer)
async def analyze_route(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_dimensions = str(img.shape)
    cv2.imwrite("temp.jpg", img)

    # line that fixed it
    encoded_img = cv2.imencode( img_dimensions,'.png')

    encoded_img = base64.b64encode(encoded_img)
   
    return{
        'filename': file.filename,
        'dimensions': img_dimensions,
        'encoded_img': encoded_img,
    }