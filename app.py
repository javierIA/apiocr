import base64
from tkinter import W
from fastapi import FastAPI, File, UploadFile,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import pandas as pd
from ocr import *
from typing import Union
from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi_simple_security import api_key_router, api_key_security


app = FastAPI()


app.include_router(api_key_router, prefix="/auth", tags=["_auth"])





class OCR(BaseModel):
    imgbox:str
    coord:str
    avatar:str


@app.post("/ocr", response_model=OCR, dependencies=[Depends(api_key_security)])
async def ocr_route(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return_img,df,avatar = inference(img)
    
    _, encoded_img = cv2.imencode('.JPG', return_img)

    encoded_image = base64.b64encode(encoded_img)
    #save txt of encoded image
    with open("temp.txt", "w") as text_file:
        text_file.write(encoded_image.decode('utf-8'))
    return{
        'imgbox': str(encoded_image),
        'coord': str(df.to_json(orient='records')),
        'avatar': str(avatar),
    }