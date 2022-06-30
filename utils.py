import cv2 
import numpy as np
from skimage.filters import threshold_local
import PIL
from PIL import Image
from PIL import ImageDraw


def draw_boxes(image, bounds):
  for (bbox, text, prob) in bounds:
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        # cleanup the text and draw the box surrounding the text along
        # with the OCR'd text itself
        text = cleanup_text(text)
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10),
          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

  return image

def grabFace(image):
  face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.3, 5):
    area = w * h
    radius = int(h * 0.75)
    cx = int(x+h/2)
    cy = int(y+w/2)
    if (area > 5000 and area < 20000):
      #cv2.circle(gray, (int(cx), int(cy)), radius, (255, 0, 255))
      crop = image[cy-radius:(cy-radius+2*radius), cx-radius:(cx-radius+2*radius)]
      #save the face
      return crop

def cleanImage(image, stage = 0):
  V = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  # applying topHat/blackHat operations
  topHat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, kernel)
  blackHat = cv2.morphologyEx(V, cv2.MORPH_BLACKHAT, kernel)
  # add and subtract between morphological operations
  add = cv2.add(V, topHat)
  subtract = cv2.subtract(add, blackHat)
  if (stage == 1):
    return subtract
  T = threshold_local(subtract, 29, offset=35, method="gaussian", mode="mirror")
  thresh = (subtract > T).astype("uint8") * 255
  if (stage == 2):
    return thresh
  # invert image 
  thresh = cv2.bitwise_not(thresh)
  return thresh

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()