import scipy.io
import torch
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from principal_RBM_aplha import *
from principal_DBN_alpha import *
from torch.utils.data import DataLoader

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


def init_DNN(sizes,nbr_classes):
    """
    un DNN est un DBN avec une couche de classification supplémentaire
    """

    DBN = init_DBN(sizes)
    n_hidden_minus_1 = DBN[-1]["W"].shape[1]
    W_classification_layer = np.random.normal(0, 0.01, (n_hidden_minus_1, nbr_classes))  # Poids de la couche de classification
    b_classification_layer = np.zeros(nbr_classes)  # Biais de la couche de classification

    # Créer le DNN
    DNN = {
        "DBN": DBN,  # Couches cachées (DBN)
        "W_classification_layer": W_classification_layer,  # Poids de la couche de classification
        "b_classification_layer": b_classification_layer  # Biais de la couche de classification
    }

    return DNN


def pretrain_DNN(X, sizes, epochs=100, learning_rate=0.01, batch_size=32, image_size=(28,28)):
    DBN_trained = train_DBN(X, sizes, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, image_size=image_size)
    DNN["DBN"] = DBN_trained["DBN"]
    return DNN

def calcul_softmax(X):
    '''
    X est supposé être un vecteur colonnes représentant l'output d'un RBM ou de la couche de classification.
    Donc de shape (nbr_units,)
    :param X:
    :return: Softmax(X)
    '''
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)


def entree_sortie_reseau(X,DNN):
    assert X.shape[1] == DNN['DBN'][0]['W'].shape[0] , "il faut avoir la même dimension d'image quelle celle sur laquelle le modèle s'est entrainé"
    data = [X] #la premiere couche est celle correspondant à l'image elle même
    for RBM in DNN['DBN']: #passe à travers les RBM
        sortie = entree_sortie_RBM(RBM, data[-1])
        data.append(sortie)
    activation_sortie = sortie@DNN["W_classification_layer"] + DNN["b_classification_layer"]  # Calcul de l'activation
    probas_sortie = calcul_softmax(activation_sortie)  # Calcul des probabilités softmax
    data.append(probas_sortie)  # Stocker les probabilités de sortie
    # Probas_sortie = probabilité pour chaque classe pour une instance donnée
    return data, probas_sortie


def calcul_entropie_croisee(probas, labels):
    """
    Calcule l'entropie croisée entre les probabilités prédites et les labels.

    Paramètres :
    - probas (numpy array) : Probabilités prédites de taille (N, C), où N est le nombre d'exemples et C le nombre de classes.
    - labels (numpy array) : Labels one-hot encodés de taille (N, C).

    Retourne :
    - entropie (float) : Valeur de l'entropie croisée.
    """
    N = probas.shape[0]  # Nombre d'exemples
    return -np.sum(labels * np.log(probas + 1e-10)) / N  # Ajout de 1e-10 pour éviter log(0)


def to_one_hot(labels, n_classes):
    """
    Convertit les labels entiers en one-hot encoding.

    Paramètres :
    - labels (numpy array) : Labels entiers de taille (N,).
    - n_classes (int) : Nombre total de classes.

    Retourne :
    - one_hot (numpy array) : Labels one-hot encodés de taille (N, n_classes).
    """
    return np.eye(n_classes)[labels]

def retropropagation(DNN, dataset, epochs, learning_rate, batch_size,image_size):
    """
    Entraîne un DNN en utilisant la rétropropagation pour minimiser l'entropie croisée.

    Paramètres :
    - DNN (dict) : Dictionnaire contenant les poids et biais du DNN.
    - X (numpy array) : Données d'entrée de taille (N, D), où N est le nombre d'exemples et D la dimension des visibles.
    - Y (numpy array) : Labels one-hot encodés de taille (N, C), où C est le nombre de classes.
    - epochs (int) : Nombre d'itérations de la descente de gradient.
    - learning_rate (float) : Taux d'apprentissage.
    - batch_size (int) : Taille du mini-batch.

    Retourne :
    - DNN (dict) : Dictionnaire contenant les poids et biais du DNN après entraînement.
    """
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, Y_batch in train_loader:

            X_batch = X_batch.numpy().reshape(batch_size, image_size[0]*image_size[1])
            Y_batch = Y_batch.numpy()
            Y_batch_one_hot = to_one_hot(Y_batch, n_classes=10)

            sorties, probas_sortie = entree_sortie_reseau(DNN = DNN, X = X_batch)
            # Calcul du gradient de l'entropie croisée par rapport à la sortie
            error = probas_sortie - Y_batch_one_hot  # Gradient de la couche de sortie

            # Rétropropagation à travers la couche de classification
            # Gradient empirique des poids de la couche de sortie, on ne veut pas la probabilité softmax mais l'entrée de la couche de classification
            # IE la sortie du dernier RBM
            dW_out = (sorties[-2].T @ error) / batch_size

            # Gradient empirique des biais de la couche de sortie, on additionne la contribution de chaque instance
            db_out = np.sum(error, axis=0) / batch_size

            # Rétropropagation à travers les couches cachées
            dA = np.dot(error, DNN["W_classification_layer"].T)
            dA *= sorties[-2] * (1 - sorties[-2])  # Dérivée de la fonction sigmoïde

            gradients = []
            for j in range(len(DNN["DBN"]) - 1, -1, -1):
                dW = np.dot(sorties[j].T, dA) / batch_size  # Gradient des poids de la couche cachée
                db = np.sum(dA, axis=0) / batch_size  # Gradient des biais de la couche cachée
                gradients.append((dW, db))

                if j > 0:
                    dA = np.dot(dA, DNN["DBN"][j]["W"].T)
                    dA *= sorties[j] * (1 - sorties[j])  # Dérivée de la fonction sigmoïde

            # Mise à jour des poids et biais de la couche de classification
            DNN["W_classification_layer"] -= learning_rate * dW_out
            DNN["b_classification_layer"] -= learning_rate * db_out

            # Mise à jour des poids et biais des couches cachées
            for j, (dW, db) in enumerate(reversed(gradients)):
                DNN["DBN"][j]["W"] -= learning_rate * dW
                DNN["DBN"][j]["b"] -= learning_rate * db

        # Calcul de l'entropie croisée à la fin de chaque époque
        X, Y= next(iter(train_loader))  # Récupère un batch
        X = X.numpy().reshape(-1, image_size[0]*image_size[1])
        Y_one_hot = to_one_hot(Y.numpy(), n_classes=10)
        _, probas_sortie = entree_sortie_reseau(X, DNN)
        entropie = calcul_entropie_croisee(probas_sortie, Y_one_hot)
        print(f"Epoch {epoch + 1}/{epochs}, Entropie croisée : {entropie:.4f}")

    return DNN


def test_DNN(dataset,DNN, image_size, batch_size =32):
    """
    Calcule le taux d'erreur
    :param dataset:
    :param DNN:
    :param batch_size:
    :return:
    """
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X, Y = next(iter(test_loader))
    X = X.numpy().reshape(batch_size, image_size[0]*image_size[1])
    Y_one_hot = to_one_hot(Y.numpy(), n_classes=10)

    data, probas_sortie  = entree_sortie_reseau(X, DNN)

    #calcule du taux d'erreur
    labels = np.argmax(probas_sortie, axis=1)
    return np.mean(labels != Y)

#définition d'une transforamtion pytorch
class BinarizeImage:
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, x):
        # Binarisation de l'image : pixels supérieurs au seuil deviennent 1, les autres deviennent 0
        return torch.where(x > self.threshold / 255.0, torch.tensor(1.0), torch.tensor(0.0))




 # Nombre de classes pour la couche de classification
parametres = {
    'n_epochs_GD_DNN' : 100,
    'n_epochs_RBM' : 100,
    'learning_rate_GD_DNN' : 0.001,
    'learning_rate_RBM' : 0.001,
    'batch_size_GD_DNN' : 32,
    'batch_size_RBM' : 32,
    'n_données' : 100,
    'n_classes_MNIST' : 10,
    'image_size_MNIST' : (28,28),
    'image_size' : (16,20),
}

parametres['taille_reseau'] =  [parametres["image_size_MNIST"][0]*parametres["image_size_MNIST"][1], 800, 500, 200,100]

transform = transforms.Compose([
    transforms.ToTensor(),
    BinarizeImage(threshold=128), #binariser blanc/noir
    transforms.Resize((parametres["image_size"][0], parametres["image_size"][1])),
])



DNN = init_DNN(parametres["taille_reseau"], parametres["n_classes_MNIST"])  # Initialisation du DNN
train_dataset = MNIST(root="datasets/raw", train=True, download=False, transform=transform)
DNN_trained = retropropagation(DNN = DNN, dataset= train_dataset, epochs=20, learning_rate=0.001, batch_size=200, image_size = parametres["image_size"])

test_dataset = MNIST(root="datasets/raw", train=False, download=False, transform=transform)
#rate_error = test_DNN(DNN=DNN, dataset = test_dataset)
