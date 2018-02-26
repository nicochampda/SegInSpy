import cv2
import numpy as np
import matplotlib.pyplot as plt

rep = '/homes/nchampda/Bureau/Sanssauvegarde/'
#img = cv2.imread(rep + 'normal/Doors/DOR_S1_1.jpg', 0)
path = "/homes/mvu/Bureau/Sanssauvegarde/"
img = cv2.imread(path + "Doors/DOR_S1_18" + ".jpg", cv2.IMREAD_COLOR)
img = cv2.imread(path + "Stairs/STR_S2_86" + ".jpg", cv2.IMREAD_COLOR)
img = cv2.imread(path + "Sign/SGN_S1_223" + ".jpg", cv2.IMREAD_COLOR)

# Conversion en niveaux de gris
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#egalisation d'histogramme
#equ = cv2.equalizeHist(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equ = clahe.apply(gray_img)

#ouverture (erosion --> dilatation)
kernel = np.ones((2,2), np.uint8) #carré
opn = cv2.morphologyEx(equ, cv2.MORPH_OPEN, kernel)

#flou gaussien
blr = cv2.blur(opn, (10, 10))

plt.hist(blr.flatten(), bins=255)
plt.show()

#seuillage
ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thr2 = cv2.threshold(equ, 127, 255, cv2.THRESH_BINARY)
ret, thr3 = cv2.threshold(opn, 127, 255, cv2.THRESH_BINARY)

ret, thr4 = cv2.threshold(blr, 18, 255, cv2.THRESH_TOZERO)
ret, thr5 = cv2.threshold(thr4, 80, 255, cv2.THRESH_BINARY_INV)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(thr5, cmap = 'gray')
plt.show()

# Contours
im2, contours, hierarchy = cv2.findContours(thr5, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)

rgb_seg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_seg)
plt.show()

#plt.subplot(241)
#plt.imshow(img, cmap = 'gray')
#plt.subplot(242)
#plt.imshow(equ, cmap = 'gray')
#plt.subplot(243)
#plt.imshow(opn, cmap = 'gray')
#plt.subplot(244)
#plt.imshow(blr, cmap = 'gray')
#plt.subplot(245)
#plt.imshow(thr1, cmap = 'gray')
#plt.subplot(246)
#plt.imshow(thr2, cmap = 'gray')
#plt.subplot(247)
#plt.imshow(thr3, cmap = 'gray')
#plt.subplot(248)
#plt.imshow(thr4, cmap = 'gray')
#plt.show()
