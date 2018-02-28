import os
import gc
import cv2
import pickle
import numpy as np
from random import sample
from sklearn.svm import LinearSVC
from preprocessing import preprocessing
from sklearn.preprocessing import StandardScaler


def main():
    """Tire un certain nombre d'images aleatoirement, effectue
    le pretraitement, effectue l'apprentissage du SVC et teste
    la classification sur un set de test
    """

    print("Apprentissage 5")
    #print("Seuils: 18 - 80")
    #print("Ouverture: 40 x 40")

    train_size = 100
    test_size  = 30

    # Porte, signe, escaliers
    # PATH a mettre a jour selon l'emplacement des images
    PATHS = ["~/Bureau/Sanssauvegarde/Doors/",
             "~/Bureau/Sanssauvegarde/Sign/",
             "~/Bureau/Sanssauvegarde/Stairs/"]
    list_dir = [os.listdir(PATHS[0]), os.listdir(PATHS[1]), os.listdir(PATHS[2])]
    try:
        list_dir[0].remove("Thumbs.db")
        list_dir[1].remove("Thumbs.db")
        list_dir[2].remove("Thumbs.db")
    except ValueError:
        pass


    # Tirage au sort des images
    train_rand_index = [None, None, None]
    test_rand_index  = [None, None, None]
    rand_index = np.random.permutation(np.arange(len(list_dir[0])) # Porte
    train_rand_index[0] = rand_index[:train_size]
    test_rand_index[0]  = rand_index[train_size:train_size + test_size]
    rand_index = np.random.permutation(np.arange(len(list_dir[1]))) # Signe
    train_rand_index[1] = rand_index[:train_size]
    test_rand_index[1]  = rand_index[train_size:train_size + test_size]
    rand_index = np.random.permutation(np.arange(len(list_dir[2]))) # Stairs
    train_rand_index[2] = rand_index[:train_size]
    test_rand_index[2]  = rand_index[train_size:train_size + test_size]

    train_X = []
    train_y = []
    # Ouverture des images d'entrainement
    for i in range(train_size):
        for j in range(len(list_dir)): # Pour chaque classe

            index_image = train_rand_index[j][i]

            # Ouverture de l'image courante
            img = cv2.imread(PATHS[j] + list_dir[j][index_image], cv2.IMREAD_GRAYSCALE)

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
    for i in range(test_size):
        for j in range(len(list_dir)): # Pour chaque classe

            index_image = test_rand_index[j][i]

            # Ouverture de l'image courante
            img = cv2.imread(PATHS[j] + list_dir[j][index_image], cv2.IMREAD_GRAYSCALE)

            # Preprocessing de l'image
            pp_img = preprocessing(img)

            # Enregistrement de l'image dans un set de test unique
            #test_X.append(pp_img.flatten())
            # Enregistrement de la classe dans le set de test
            # 0 : porte, 1 : signe, 2 : escaliers
            #test_y.append(j)

            # Enregistrement dans un set de test pour chaque classe
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
    print("Train X :", train_X.shape)
    print(" Test X :", test_X_porte.shape)

    # Normalisation
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X_porte = scaler.transform(test_X_porte)
    test_X_signe = scaler.transform(test_X_signe)
    test_X_stair = scaler.transform(test_X_stair)
    # Sauvegarde des parametres de normalisation
    #pickle.dump(scaler, open("scaler_svm_t5.sav", 'wb'))
    print("Scaling termine")
    gc.collect()

    # Entrainement
    clf = LinearSVC()
    clf.fit(train_X, train_y)
    print("Training termine")
    # Sauvegarde du modele svc
    #pickle.dump(clf, open("model_svm_t5.sav", 'wb'))
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

