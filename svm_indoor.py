import os
import gc
import cv2
import pickle
import numpy as np
from preprocessing import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def main():
    """docstring for main"""

    train_size = 100
    test_size  = 30

    # Porte, signe, escaliers
    PATHS = ["/homes/mvu/Bureau/Sanssauvegarde/Doors/",
            "/homes/mvu/Bureau/Sanssauvegarde/Sign/",
            "/homes/mvu/Bureau/Sanssauvegarde/Stairs/"]
    list_dir = [os.listdir(PATHS[0]), os.listdir(PATHS[1]), os.listdir(PATHS[2])]

    train_X = []
    train_y = []
    # Ouverture des images d'entrainement
    for i in range(train_size):
        #print("Entrainement:", i)
        for j in range(len(list_dir)): # Pour chaque classe

            # Ouverture de l'image courante
            img = cv2.imread(PATHS[j] + list_dir[j][i], cv2.IMREAD_GRAYSCALE)

            # Preprocessing de l'image
            pp_img = preprocessing(img)

            # Enregistrement de l'image dans le set d'entrainement
            train_X.append(pp_img.flatten())
            # Enregistrement de la classe dans le set d'entrainement
            # 0 : porte, 1 : signe, 2 : escaliers
            train_y.append(j)


    test_X_porte = []
    test_y_porte = np.zeros(test_size)
    test_X_signe = []
    test_y_signe = np.ones(test_size)
    test_X_stair = []
    test_y_stair = np.ones(test_size) * 2
    # Ouverture des images de test
    for i in range(train_size, train_size + test_size):
        #print("Test:", i)
        for j in range(len(list_dir)): # Pour chaque classe

            # Ouverture de l'image courante
            img = cv2.imread(PATHS[j] + list_dir[j][i], cv2.IMREAD_GRAYSCALE)

            # Preprocessing de l'image
            pp_img = preprocessing(img)

            # Enregistrement de l'image dans le set de test
            #test_X.append(pp_img.flatten())
            # Enregistrement de la classe dans le set de test
            # 0 : porte, 1 : signe, 2 : escaliers
            #test_y.append(j)

            if j == 0:
                test_X_porte.append(pp_img.flatten())
            elif j == 1:
                test_X_signe.append(pp_img.flatten())
            elif j == 2:
                test_X_stair.append(pp_img.flatten())

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    test_X_porte = np.array(test_X_porte)
    test_y_porte = np.array(test_y_porte)
    test_X_signe = np.array(test_X_signe)
    test_y_signe = np.array(test_y_signe)
    test_X_stair = np.array(test_X_stair)
    test_y_stair = np.array(test_y_stair)

    print("Ouverture des images terminee")
    print("Test X porte:", test_X_porte.shape)
    print("Test X signe:", test_X_signe.shape)
    print("Test X stair:", test_X_stair.shape)

    # Entrainement
    # Scaling
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X_porte = scaler.transform(test_X_porte)
    test_X_signe = scaler.transform(test_X_signe)
    test_X_stair = scaler.transform(test_X_stair)
    print("Scaling termine")
    gc.collect()

    clf = LinearSVC()
    clf.fit(train_X, train_y)
    print("Training termine")
    filename = "model_svm.sav"
    pickle.dump(clf, open(filename, 'wb'))
    print("Enregistrement termine")

    # Resultat sur les images de test
    score_porte = clf.score(test_X_porte, test_y_porte)
    score_signe = clf.score(test_X_signe, test_y_signe)
    score_stair = clf.score(test_X_stair, test_y_stair)

    print("Score porte:", score_porte)
    print("Score signe:", score_signe)
    print("Score stair:", score_stair)


if __name__ == '__main__':
    main()

