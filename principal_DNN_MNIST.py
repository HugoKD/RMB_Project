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
from torch.utils.data import DataLoader, Dataset

#raw_dataset = MNIST(root="datasets/", train=True, download=True)
#Dataset organisé comme 36 classes chacune contenant ~30 images A-Z + 0-9


def lire_alpha_digit(data, indices_classes):

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

    DBN = init_DBN(sizes)
    n_hidden_minus_1 = DBN[-1]["W"].shape[1] #couche de sortie du DBN
    W_classification_layer = np.random.normal(0, 0.01, (n_hidden_minus_1, nbr_classes))  # Poids de la couche de classification
    b_classification_layer = np.zeros(nbr_classes)  # Biais de la couche de classification

    # Créer le DNN
    DNN = {
        "DBN": DBN,  # Couches cachées (DBN)
        "W_classification_layer": W_classification_layer,  # Poids de la couche de classification
        "b_classification_layer": b_classification_layer  # Biais de la couche de classification
    }

    return DNN


def pretrain_DNN(dataset, DNN, sizes, epochs, learning_rate, batch_size, image_size, plot):
    DBN_trained = train_DBN(dataset, sizes, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, image_size=image_size, plot = plot)
    DNN["DBN"] = DBN_trained #on remplace celui déjà instencié ?
    return DNN

import torch

def calcul_softmax(X):
    # Pour améliorer la stabilité numérique, on soustrait le max par ligne
    if isinstance(X, torch.Tensor):
        X_max = torch.max(X, dim=1, keepdim=True)[0]
        exp_X = torch.exp(X - X_max)
        return exp_X / torch.sum(exp_X, dim=1, keepdim=True)
    else:
        X_max = np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X - X_max)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

import numpy as np

def entree_sortie_reseau(X, DNN):
    # Vérifie que la dimension de l'image correspond à celle du modèle
    assert X.shape[1] == DNN['DBN'][0]['W'].shape[0], \
        "La dimension de l'image doit être identique à celle sur laquelle le modèle a été entraîné"

    data = [X]  # Première couche : l'image d'entrée
    v = X.copy()

    # Propagation stochastique à travers les RBMs (les couches cachées)
    for RBM in DNN['DBN']:
        p_h = entree_sortie_RBM(RBM, v)  # Probabilités d'activation
        v = np.random.binomial(1, p_h)   # Échantillonnage binaire
        data.append(p_h)  # On stocke les probabilités pour analyse ou debugging

    # Calcul de l'activation de la couche de classification
    activation_sortie = v @ DNN["W_classification_layer"] + DNN["b_classification_layer"]
    probas_sortie = calcul_softmax(activation_sortie)  # Probabilités softmax en sortie
    data.append(probas_sortie)

    return data, probas_sortie



def calcul_entropie_croisee(probas, labels):
    """
    Calcule l'entropie croisée entre les probabilités de sortie et les labels one-hot.
    """
    N = probas.shape[0]  # Nombre d'exemples
    return -np.sum(labels * np.log(probas + 1e-10)) / N  # 1e-10 pour éviter log(0)

def to_one_hot(labels, n_classes):
    """
    Convertit un vecteur de labels en représentation one-hot.
    """
    return np.eye(n_classes)[labels]

def retropropagation(DNN, dataset, epochs, learning_rate, batch_size, image_size, n_classes,plot = False):
    """
    Entraîne un DNN (DBN empilé avec une couche de classification) en utilisant la rétropropagation
    pour minimiser l'entropie croisée.
    """
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    entropies = []
    for epoch in range(epochs):
        for X_batch, Y_batch in train_loader:
            # Taille effective du batch (utile pour le dernier batch possiblement incomplet)
            current_batch_size = X_batch.shape[0]
            X_batch = X_batch.numpy().reshape(current_batch_size, image_size[0] * image_size[1])
            Y_batch = Y_batch.numpy()

            # Conversion des labels en one-hot avec le bon nombre de classes
            Y_batch_one_hot = to_one_hot(Y_batch, n_classes=n_classes)

            # Propagation avant
            sorties, probas_sortie = entree_sortie_reseau(X_batch, DNN)

            # Calcul de l'erreur (gradient de l'entropie croisée)
            error = probas_sortie - Y_batch_one_hot

            # Rétropropagation à travers la couche de classification
            # On utilise la sortie du dernier RBM (avant softmax) pour calculer les gradients
            dW_out = (sorties[-2].T @ error) / current_batch_size
            db_out = np.sum(error, axis=0) / current_batch_size

            # Rétropropagation dans les couches cachées
            dA = error @ DNN["W_classification_layer"].T
            dA *= sorties[-2] * (1 - sorties[-2])  # Dérivée de la sigmoïde

            gradients = []
            # Parcours des couches cachées en sens inverse
            for j in range(len(DNN["DBN"]) - 1, -1, -1):
                dW = (sorties[j].T @ dA) / current_batch_size
                db = np.sum(dA, axis=0) / current_batch_size
                gradients.append((dW, db))

                # Si ce n'est pas la première couche d'entrée, rétropropager l'erreur
                if j > 0:
                    dA = dA @ DNN["DBN"][j]["W"].T
                    dA *= sorties[j] * (1 - sorties[j])

            # Mise à jour des poids et biais de la couche de classification
            DNN["W_classification_layer"] -= learning_rate * dW_out
            DNN["b_classification_layer"] -= learning_rate * db_out

            # Mise à jour des poids et biais des couches cachées
            for j, (dW, db) in enumerate(reversed(gradients)):
                DNN["DBN"][j]["W"] -= learning_rate * dW
                DNN["DBN"][j]["b"] -= learning_rate * db

        # Calcul de l'entropie croisée sur un batch de validation à la fin de chaque époque
        X_val, Y_val = next(iter(train_loader))
        current_val_batch = X_val.shape[0]
        X_val = X_val.numpy().reshape(current_val_batch, image_size[0] * image_size[1])
        Y_val_one_hot = to_one_hot(Y_val.numpy(), n_classes=n_classes)
        _, probas_val = entree_sortie_reseau(X_val, DNN)
        entropie = calcul_entropie_croisee(probas_val, Y_val_one_hot)
        entropies.append(entropie)
        print(f"Epoch {epoch + 1}/{epochs}, Entropie croisée : {entropie:.4f}")

    if plot:
        plt.plot(range(1, epochs + 1), entropies, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Entropie croisée")
        plt.title("Évolution de l'entropie croisée pendant l'entraînement")
        plt.grid(True)
        plt.show()

    return DNN


def DNN_test(dataset, DNN, true_labels):
    """
    Évalue le modèle sur un dataset en calculant le taux d'erreur et l'exactitude.

    :param dataset: Dataset de test
    :param DNN: Modèle DNN (structure contenant la DBN et la couche de classification)
    :param true_labels: Étiquettes vraies sous forme d'un tableau NumPy
    :return: exactitude et taux d'erreur
    """
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(test_loader))
    x, y = data[0], data[1]
    if hasattr(x, "numpy"):
        x = x.numpy()
    pred_labels = np.argmax(entree_sortie_reseau(DNN=DNN, X=x)[1], axis=1)
    accuracy = np.mean(pred_labels == np.array(true_labels))
    return 1 - accuracy


##################################################################################################
################################### Parties 4 & 5 ################################################
##################################################################################################


################################# Juste pour le RBM ##############################################
