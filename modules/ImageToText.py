from PIL import Image
import pytesseract as pt 
import numpy as np
import cv2
import re

pt.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def imageToText(filename):
    pattern = "[^\x00-\x7F]+"
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray,3)
    text = pt.image_to_string(image)
    text = text.strip().replace("\n","")
    text = re.sub(pattern,"",text)
    return text

