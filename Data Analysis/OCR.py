from PIL import Image
import pytesseract as pt
import argparse
import cv2
import os


pt.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
#file:///C:/Program Files/Tesseract-OCR/tesseract.exe
# load the example image and convert it to grayscale
image = cv2.imread("C:/Users/DELL/Pictures/Screenshots/Screenshot (45).png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess the
# image
#if args["preprocess"] == "thresh":
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# make a check to see if median blurring should be done to remove
# noise
#elif args["preprocess"] == "blur":
gray = cv2.medianBlur(gray, 3)
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
#filename = "ocr_processed.png"
#cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pt.image_to_string(image)
#os.remove(filename)
print(text)

# show the output images
#cv2.imshow("Image", image)
#cv2.imshow("Output", gray)
#cv2.waitKey(0)