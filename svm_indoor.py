import os
import cv2
import numpy

def main():
    """docstring for main"""

    train_size = 100
    test_size  = 30

    # Porte, signe, escaliers
    PATHS = ["", "", ""]
    list_dir = [os.listdir(PATHS[0]), os.listdir(PATHS[1]), os.listdir(PATHS[2])]

    train_X = []
    train_y = []
    # Ouverture des images d'entrainement
    for i in range(train_size):
        for j in range(len(list_dir)): # Pour chaque classe

            # Ouverture de l'image courante
            img = cv2.imread(PATHS[j] + list_dir[j][i], cv2.IMREAD_GRAYSCALE)

            # Preprocessing de l'image
            pp_img = preprocessing(img)

            # Enregistrement de l'image dans le set d'entrainement
            train_X.append(pp_img)
            # Enregistrement de la classe dans le set d'entrainement
            # 0 : porte, 1 : signe, 2 : escaliers
            train_y.append(j)



if __name__ == '__main__':
    main()

