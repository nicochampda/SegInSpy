import cv2
import numpy as np

def preprocessing(img):
    #resize
    small_img = cv2.resize(img, (500,500))

    #egalisation d'histogrammes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(small_img)
    
    #flou gaussien
    blr = cv2.medianBlur(equ, 9)
    
    #seuillage
    ret, thr4 = cv2.threshold(blr, 20, 255, cv2.THRESH_TOZERO)
    ret, thr5 = cv2.threshold(thr4, 80, 255, cv2.THRESH_BINARY_INV)
    
   
    #ouverture (erosion --> dilatation)
    kernel = np.ones((10,10), np.uint8) #carré
    opn = cv2.morphologyEx(thr5, cv2.MORPH_OPEN, kernel)
    
    return opn


def main():
    """Fonction de test de la fonction de preprocessing"""

    # Ouverture de l'image
    img = cv2.imread("DOR_S1_38.jpg", cv2.IMREAD_GRAYSCALE)

    # Pretraitement
    pp_img = preprocessing(img)

    plt.subplot(121)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(122)
    plt.imshow(pp_img, cmap = 'gray')
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
