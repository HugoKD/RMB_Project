import scipy.io
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import os
import matplotlib.pyplot as plt
import numpy as np
from principal_RBM_aplha import *

file_path = os.path.join("datasets", "binaryalphadigs.mat")
data = scipy.io.loadmat(file_path)
images = data['dat']
labels = data['classlabels']

#raw_dataset = MNIST(root="datasets/", train=True, download=True)
#Dataset organisé comme 36 classes chacune contenant ~30 images A-Z + 0-9
def show_image(class_idx, sample_idx):
    img = np.array(images[class_idx][sample_idx], dtype=np.uint8)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def lire_alpha_digit(data, indices_classes):
    """
    Charge les caractères sélectionnés depuis le dataset Binary AlphaDigits et les transforme en matrice.

    Paramètres :
    - file_path (str) : Chemin du fichier .mat contenant le dataset.
    - indices_classes (list) : Liste des indices des caractères à récupérer (0-35).

    Retourne :
    - X (numpy array) : Matrice des données (N, D), où N = nombre d'images et D = 320 (20x16 pixels aplatis).
    """

    X = []
    images = data['dat']
    labels = data['classlabels']

    for idx in indices_classes:
        for img in images[idx]:  # Parcourir les 39 images de la classe choisie
            X.append(img.flatten())  # Transformer l'image en vecteur (1D)

    # Conversion en array numpy
    X = np.array(X, dtype=np.float32)

    return X



def init_DBN(sizes):
    """
    Initialise un Deep Belief Network (DBN) en empilant plusieurs RBMs.

    Paramètres :
    - sizes (list) : Liste des tailles des couches du DBN [n_visible_1,n_cachée_1,n_visible_2, n_cachée_2, n_visible_3,n_cachée_3,...n_visible_n,n_cachée_n]
    Où n = nombre de RBM = nombre de couches
    Le premier element de la liste correspond à la taille de l'image = n_visible_1
    Retourne :
    - DBN (list) : Liste de dictionnaires représentant les RBMs empilés.
                   Chaque dictionnaire contient les poids et biais d'un RBM.
    """
    DBN = []  # Liste pour stocker les RBMs
    # Parcourir les couches pour initialiser les RBMs
    for i in range(len(sizes) - 1):
        n_visible = sizes[i]      # Nombre de neurones visibles (couche actuelle)
        n_hidden = sizes[i + 1]   # Nombre de neurones cachés (couche suivante)

        # Initialiser un RBM pour cette paire de couches
        rbm = init_RBM(n_visible, n_hidden)

        # Ajouter le RBM au DBN
        DBN.append(rbm)

    return DBN



def train_DBN(X, sizes, epochs=100, learning_rate=0.01, batch_size=32, image_size = (16,20)):
    """
    Entraîne un DBN de manière non supervisée en utilisant la procédure Greedy Layer-Wise Training.

    Paramètres :
    - X (numpy array) : Données d'entrée de taille (N, D), où N est le nombre d'exemples et D la dimension des visibles.
    - sizes (list) : Liste des tailles des couches du DBN.
    - epochs (int) : Nombre d'itérations de la descente de gradient pour chaque RBM.
    - learning_rate (float) : Taux d'apprentissage pour chaque RBM.
    - batch_size (int) : Taille du mini-batch pour chaque RBM.

    Retourne :
    - DBN (list) : Liste de dictionnaires représentant les RBMs pré-entraînés.
    """
    assert image_size[0] * image_size[1] == sizes[0], "La première couche doit correspondre à la taille flatten de l'image"
    DBN = init_DBN(sizes)  # Initialisation du DBN

    # Données d'entrée pour la première couche
    input_data = X

    # Entraînement couche par couche
    for i in range(len(DBN)):
        print(f"Entraînement de la couche {i + 1}/{len(DBN)}...")

        # Entraînement du RBM actuel
        DBN[i] = train_RBM(
            input_data,
            n_hidden=sizes[i + 1],
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )

        # Propagation des données à travers le RBM entraîné
        if i < len(DBN) - 1:  # Ne pas propager pour la dernière couche
            input_data = entree_sortie_RBM(DBN[i], input_data)

    print("Pré-entraînement du DBN terminé !")
    return DBN


def generer_image_DBN(DBN, n_iterations=100, n_images=10, image_shape=(20, 16), plot = True):

    n_visible = DBN[0]["a"].shape[0]  # Dimension de l'image 20*16 = 320

    generated_images = np.random.rand(n_images, n_visible)
    #Propager cette images dans le RBN et puis la faire revenir
    for _ in range(n_iterations):
        for rbm in DBN:
            generated_images = entree_sortie_RBM(rbm, generated_images)
            generated_images = (np.random.rand(*generated_images.shape) < generated_images).astype(np.float32)

            # Propagation descendante à travers les couches du DBN
        for rbm in reversed(DBN):
            generated_images = sortie_entree_RBM(rbm, generated_images)
            generated_images = (np.random.rand(*generated_images.shape) < generated_images).astype(np.float32)

    generated_images = generated_images.reshape(n_images, *image_shape)

    # Afficher les images générées si plot est True
    if plot:
        plt.figure(figsize=(10, 4))
        for i in range(n_images):
            plt.subplot(1, n_images, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Images générées par le DBN après {n_iterations} itérations de Gibbs et 500 epochs d'entrainement")
        plt.show()

    return generated_images

'''
sizes = [320,784, 500, 200]  # Tailles des couches du DBN
X = lire_alpha_digit(data, indices_classes=[10])  # Charger les données
DBN_trained = train_DBN(X, sizes, epochs=500, learning_rate=0.01, batch_size=32)
generated_images = generer_image_DBN(DBN_trained, n_iterations=100, n_images=10, image_shape=(20, 16), plot=True)


# Affichage des poids et biais du premier RBM entraîné
print("Premier RBM entraîné :")
print("W :", DBN_trained[0]["W"].shape)  # Poids (784, 500)
print("a :", DBN_trained[0]["a"].shape)  # Biais visibles (784,)
print("b :", DBN_trained[0]["b"].shape)  # Biais cachés (500,)

# Affichage des poids et biais du deuxième RBM entraîné
print("\nDeuxième RBM entraîné :")
print("W :", DBN_trained[1]["W"].shape)  # Poids (500, 200)
print("a :", DBN_trained[1]["a"].shape)  # Biais visibles (500,)
print("b :", DBN_trained[1]["b"].shape)  # Biais cachés (200,)

print("\n3 RBM entraîné :")
print("W :", DBN_trained[2]["W"].shape)  # Poids (500, 200)
print("a :", DBN_trained[2]["a"].shape)  # Biais visibles (500,)
print("b :", DBN_trained[2]["b"].shape)  # Biais cachés (200,)
'''
