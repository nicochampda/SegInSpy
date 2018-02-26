import cv2
import numpy as np

def preprocessing(img):
    #resize
    small_img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    
    #egalisation d'histogrammes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(small_img)
    
    #flou gaussien
    blr = cv2.blur(equ, (10, 10))
    
    #seuillage
    ret, thr4 = cv2.threshold(blr, 18, 255, cv2.THRESH_TOZERO)
    ret, thr5 = cv2.threshold(thr4, 80, 255, cv2.THRESH_BINARY_INV)

    #ouverture (erosion --> dilatation)
    kernel = np.ones((50,50), np.uint8) #carré
    opn = cv2.morphologyEx(thr5, cv2.MORPH_OPEN, kernel)
    
    return opn
    