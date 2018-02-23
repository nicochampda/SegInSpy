import cv2
import numpy as np
import matplotlib.pyplot as plt

rep = '/homes/nchampda/Bureau/Sanssauvegarde/'
img = cv2.imread(rep + 'normal/Doors/DOR_S1_1.jpg', 0)

#egalisation d'histogramme
#equ = cv2.equalizeHist(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equ = clahe.apply(img)

#ouverture (erosion --> dilatation)
kernel = np.ones((20,20), np.uint8) #carré
opn = cv2.morphologyEx(equ, cv2.MORPH_OPEN, kernel)

#flou gaussien
blr = cv2.blur(opn, (10, 10))

#seuillage
ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thr2 = cv2.threshold(equ, 127, 255, cv2.THRESH_BINARY)
ret, thr3 = cv2.threshold(opn, 127, 255, cv2.THRESH_BINARY)
ret, thr4 = cv2.threshold(blr, 127, 255, cv2.THRESH_BINARY)

plt.subplot(241)
plt.imshow(img, cmap = 'gray')
plt.subplot(242)
plt.imshow(equ, cmap = 'gray')
plt.subplot(243)
plt.imshow(opn, cmap = 'gray')
plt.subplot(244)
plt.imshow(blr, cmap = 'gray')
plt.subplot(245)
plt.imshow(thr1, cmap = 'gray')
plt.subplot(246)
plt.imshow(thr2, cmap = 'gray')
plt.subplot(247)
plt.imshow(thr3, cmap = 'gray')
plt.subplot(248)
plt.imshow(thr4, cmap = 'gray')
plt.show()
