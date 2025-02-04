import scipy.io
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np


file_path = os.path.join("datasets", "binaryalphadigs.mat")
data = scipy.io.loadmat(file_path)
print(data.keys())
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




def init_RBM(n_visible, n_hidden):
    """
    Initialise un RBM avec des poids et des biais.

    Paramètres :
    - n_visible (int) : Nombre de neurones visibles (dimension des données).
    - n_hidden (int) : Nombre de neurones cachés (taille de la représentation latente).

    Retourne :
    - RBM (dict) : Dictionnaire contenant les poids et biais du modèle.
    """

    RBM = {
        "W": np.random.normal(0, 0.01, (n_visible, n_hidden)),  # Poids (matrice de taille D x H)
        "a": np.zeros(n_visible),  # Biais des neurones visibles (vecteur de taille D)
        "b": np.zeros(n_hidden)   # Biais des neurones cachés (vecteur de taille H)
    }

    return RBM




def sigmoid(x):
    """ Fonction sigmoïde. """
    return 1 / (1 + np.exp(-x))


def entree_sortie_RBM(RBM, X):
    """
    Calcule les activations des neurones cachés à partir des neurones visibles.

    Paramètres :
    - RBM (dict) : Structure contenant W (poids), a (biais visible) et b (biais caché).
    - X (numpy array) : Données d'entrée de taille (N, D), où N est le nombre d'exemples et D la dimension des visibles.

    Retourne :
    - H (numpy array) : Probabilités des neurones cachés activés (N, H).
    """

    W = RBM["W"]  # Matrice des poids (D, H)
    b = RBM["b"]  # Biais des neurones cachés (H,)

    activation = X@W + b  # Produit matriciel + biais cachés
    H = sigmoid(activation)  # Application de la fonction sigmoïde

    return H


def sortie_entree_RBM(RBM, H):
    """
    Reconstruit les activations des neurones visibles à partir des neurones cachés.

    Paramètres :
    - RBM (dict) : Contient W (poids), a (biais visible) et b (biais caché).
    - H (numpy array) : Activations des neurones cachés (N, H), où N est le nombre d'exemples.

    Retourne :
    - X_reconstruit (numpy array) : Probabilités des neurones visibles activés (N, D).
    """

    W = RBM["W"]  # Matrice des poids (D, H)
    a = RBM["a"]  # Biais des neurones visibles (D,)

    activation = np.dot(H, W.T) + a  # Produit matriciel + biais visibles
    X_reconstruit = sigmoid(activation)  # Application de la fonction sigmoïde

    return X_reconstruit


def train_RBM(X, n_hidden, epochs=100, learning_rate=0.01, batch_size=32):
    """
    Entraîne un RBM en utilisant l'algorithme de Contrastive Divergence-1 (CD-1).
    But : Apprendre une représentation efficace des images
    Méthode : Réduire la différence entre la phase positive (entrée -> sorties) et la phase negative (entrée-> sorties -> entrées -> sorties).


    Paramètres :
    - X (numpy array) : Données d'entrée de taille (N, D), où N est le nombre d'exemples et D la dimension des visibles.
    - n_hidden (int) : Nombre de neurones cachés.
    - epochs (int) : Nombre d'itérations de la méthode du gradient.
    - learning_rate (float) : Taux d'apprentissage.
    - batch_size (int) : Taille du mini-batch.

    Retourne :
    - RBM (dict) : Dictionnaire contenant les poids et biais du modèle après entraînement.
    """

    n_visible = X.shape[1]  # nbr_image_class, nbr neurones visibles = dim image

    RBM = init_RBM(n_visible, n_hidden)  # Initialisation du RBM

    n_samples = X.shape[0]  # Nombre total d'échantillons

    for epoch in range(epochs):
        # Mélanger les données à chaque époque
        np.random.shuffle(X)

        # Parcourir les mini-batches
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i + batch_size]

            # Étape positive : calcul des probabilités des neurones cachés
            H_pos = entree_sortie_RBM(RBM, X_batch)
            # Output proba des hidden units (0,1), pour chaque images (donc de shape (batch_size,n_hidden))
            #pour travailler avec des bianires plutot que des proba
            H_sample = (np.random.rand(*H_pos.shape) < H_pos).astype(np.float32)
            # Étape négative : reconstruction des visibles à partir des cachés
            X_neg = sortie_entree_RBM(RBM, H_sample)

            # Calcul des probabilités des neurones cachés pour les données reconstruites
            H_neg = entree_sortie_RBM(RBM, X_neg)

            # Mise à jour des poids et des biais
            grad_W = (np.dot(X_batch.T, H_pos) - np.dot(X_neg.T, H_neg))/batch_size
            grad_a = (np.sum(X_batch - X_neg, axis=0)) / batch_size
            grad_b = (np.sum(H_pos - H_neg, axis=0)) / batch_size

            RBM["W"] += learning_rate * grad_W
            RBM["a"] += learning_rate * grad_a
            RBM["b"] += learning_rate * grad_b

        # Calcul de l'erreur quadratique moyenne (MSE) à la fin de chaque époque
        X_reconstruit = sortie_entree_RBM(RBM, entree_sortie_RBM(RBM, X))
        mse = np.mean((X - X_reconstruit) ** 2)

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse:.4f}")

    return RBM

def generer_image_RBM(RBM, n_iterations=100, n_images=10, image_shape=(20, 16), plot = True):
    n_visible = RBM["a"].shape[0]  # Dimension de l'image 20*16 = 320

    generated_images = np.random.rand(n_images, n_visible)

    for _ in range(n_iterations):
        H = entree_sortie_RBM(RBM, generated_images)
        H_sample = (np.random.rand(*H.shape) < H).astype(np.float32)

        X_reconstruit = sortie_entree_RBM(RBM, H_sample)
        generated_images = (np.random.rand(*X_reconstruit.shape) < X_reconstruit).astype(np.float32)

    generated_images = generated_images.reshape(n_images, *image_shape)
    if plot :
        plt.figure(figsize=(10, 4))
        for i in range(n_images):
            plt.subplot(1, n_images, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Images générées après {n_iterations} itérations de Gibbs pour la classe {10} ")
        plt.show()

    return generated_images


'''
X = lire_alpha_digit(data, indices_classes=[10])  # Charger les données
RBM_trained = train_RBM(X, n_hidden=200, epochs=1000, learning_rate=0.01, batch_size=10)
generated_images = generer_image_RBM(RBM_trained, n_iterations=1000, n_images=10, image_shape=(20, 16))
'''