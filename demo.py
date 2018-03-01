import os
import gc
import cv2
import pickle
import numpy as np
from random import sample
from sklearn.svm import LinearSVC
from preprocessing import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    """docstring for main"""

    test_size  = 3

    # Porte, signe, escaliers
    # PATH a mettre a jour selon l'emplacement des images
    PATHS = ["/homes/mvu/Bureau/Sanssauvegarde/Doors/",
             "/homes/mvu/Bureau/Sanssauvegarde/Sign/",
             "/homes/mvu/Bureau/Sanssauvegarde/Stairs/"]


    test_X_pp  = []
    test_X_raw = []

    images_name = ["SGN_S2_23.jpg", "STR_S3_16.JPG", "DOR_S2_60.jpg", "SGN_S1_78.jpg", "STR_S4_159.jpg", "DOR_S3_94.JPG",
                    "DOR_S1_75.jpg", "STR_S4_194.jpg", "SGN_S1_210.jpg",
                    "SGN_S2_107.jpg"]
    test_y = [1, 2, 0, 1, 2, 0, 0, 2, 1, 1]

    # Ouverture des images de test
    for i in range(len(images_name)):

        name = images_name[i]
        j = test_y[i]

        # Ouverture de l'image courante
        img = cv2.imread(PATHS[j] + name, cv2.IMREAD_GRAYSCALE)
        small_img = cv2.resize(img, (500,500))
        test_X_raw.append(small_img.flatten())

        # Preprocessing de l'image
        pp_img = preprocessing(img)

        # Enregistrement dans un set de test pour chaque classe
        test_X_pp.append(pp_img.flatten())
        test_y.append(j)

    test_X_raw = np.array(test_X_raw)
    test_X_pp  = np.array(test_X_pp)

    print("Ouverture des images terminee")
    print("Test X :", test_X_pp.shape)

    # Ouverture des parametres de normalisation
    print("Chargement du scaler")
    scaler = pickle.load(open("models/scaler_pp.sav", 'rb'))
    scaler.transform(test_X_pp)

    # Test
    print("Chargement du modele SVM")
    clf = pickle.load(open("models/svm_pp.sav", 'rb'))
    y_pred_pp = clf.predict(test_X_pp)

    #print("y pred", y_pred_pp)
    #print("y gt  ", test_y)

    labels = ["Porte", "Signe", "Escaliers"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(y_pred_pp)):

        img_raw = cv2.cvtColor(test_X_raw[i].reshape(500, 500), cv2.COLOR_GRAY2BGR)
        img_pp  = cv2.cvtColor(test_X_pp[i].reshape(500, 500), cv2.COLOR_GRAY2BGR)

        color = (0,0, 255) # Bleu
        cv2.putText(img_raw, labels[test_y[i]], (200,250), font, 2, color, 2, cv2.LINE_AA)

        if y_pred_pp[i] == test_y[i]:
            color = (0,255,0) # Vert 
        else:
            color = (255,0,0) # Rouge 

        cv2.putText(img_pp, labels[y_pred_pp[i]], (200,250), font, 2, color, 2, cv2.LINE_AA)

        plt.subplot(121)
        plt.imshow(img_raw)
        plt.subplot(122)
        plt.imshow(img_pp)
        plt.show()


def main_random():
    """docstring for main"""

    test_size  = 3

    # Porte, signe, escaliers
    # PATH a mettre a jour selon l'emplacement des images
    PATHS = ["/homes/mvu/Bureau/Sanssauvegarde/Doors/",
             "/homes/mvu/Bureau/Sanssauvegarde/Sign/",
             "/homes/mvu/Bureau/Sanssauvegarde/Stairs/"]
    list_dir = [os.listdir(PATHS[0]), os.listdir(PATHS[1]), os.listdir(PATHS[2])]
    try:
        list_dir[0].remove("Thumbs.db")
        list_dir[1].remove("Thumbs.db")
        list_dir[2].remove("Thumbs.db")
    except ValueError:
        pass


    # Tirage au sort des images
    test_rand_index  = [None, None, None]
    rand_index = np.random.permutation(np.arange(len(list_dir[0]))) # Porte
    test_rand_index[0]  = rand_index[:test_size]
    rand_index = np.random.permutation(np.arange(len(list_dir[1]))) # Signe
    test_rand_index[1]  = rand_index[:test_size]
    rand_index = np.random.permutation(np.arange(len(list_dir[2]))) # Stairs
    test_rand_index[2]  = rand_index[:test_size]

    test_X_pp  = []
    test_X_raw = []
    test_y = []
    # Ouverture des images de test
    for i in range(test_size):
        for j in range(len(list_dir)): # Pour chaque classe

            index_image = test_rand_index[j][i]

            print(list_dir[j][index_image])

            # Ouverture de l'image courante
            img = cv2.imread(PATHS[j] + list_dir[j][index_image], cv2.IMREAD_GRAYSCALE)
            small_img = cv2.resize(img, (500,500))
            test_X_raw.append(small_img.flatten())

            # Preprocessing de l'image
            pp_img = preprocessing(img)

            # Enregistrement dans un set de test pour chaque classe
            test_X_pp.append(pp_img.flatten())
            test_y.append(j)

    test_X_raw = np.array(test_X_raw)
    test_X_pp  = np.array(test_X_pp)

    print("Ouverture des images terminee")
    print("Test X :", test_X_pp.shape)

    # Ouverture des parametres de normalisation
    print("Chargement du scaler")
    scaler = pickle.load(open("models/scaler_pp.sav", 'rb'))
    scaler.transform(test_X_pp)

    # Test
    print("Chargement du modele SVM")
    clf = pickle.load(open("models/svm_pp.sav", 'rb'))
    y_pred_pp = clf.predict(test_X_pp)

    #print("y pred", y_pred_pp)
    #print("y gt  ", test_y)

    labels = ["Porte", "Signe", "Escaliers"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(y_pred_pp)):

        img_raw = cv2.cvtColor(test_X_raw[i].reshape(500, 500), cv2.COLOR_GRAY2BGR)
        img_pp  = cv2.cvtColor(test_X_pp[i].reshape(500, 500), cv2.COLOR_GRAY2BGR)

        color = (0,0, 255) # Bleu
        cv2.putText(img_raw, labels[test_y[i]], (200,250), font, 2, color, 2, cv2.LINE_AA)

        if y_pred_pp[i] == test_y[i]:
            color = (0,255,0) # Vert 
        else:
            color = (255,0,0) # Rouge 

        cv2.putText(img_pp, labels[y_pred_pp[i]], (200,250), font, 2, color, 2, cv2.LINE_AA)

        plt.subplot(121)
        plt.imshow(img_raw)
        plt.subplot(122)
        plt.imshow(img_pp)
        plt.show()

if __name__ == '__main__':
    main()

