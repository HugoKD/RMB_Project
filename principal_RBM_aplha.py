from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch



#raw_dataset = MNIST(root="datasets/", train=True, download=True)
#Dataset organisé comme 36 classes chacune contenant ~30 images A-Z + 0-9
def show_image(class_idx, sample_idx):
    file_path = os.path.join("datasets", "binaryalphadigs.mat")
    data = scipy.io.loadmat(file_path)
    images = data['dat']
    labels = data['classlabels']
    img = np.array(images[class_idx][sample_idx], dtype=np.uint8)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def lire_alpha_digit(data, indices_classes):
    """
    Charge les caractères sélectionnés depuis le dataset Binary AlphaDigits et les transforme en matrice.
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


    RBM = {
        "W": np.random.normal(0, 0.01, (n_visible, n_hidden)),  # Poids (matrice de taille D x H)
        "a": np.zeros(n_visible),  # Biais des neurones visibles (vecteur de taille D)
        "b": np.zeros(n_hidden)   # Biais des neurones cachés (vecteur de taille H)
    }

    return RBM




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def entree_sortie_RBM(RBM, X):


    W = RBM["W"]  # Matrice des poids (D, H)
    b = RBM["b"]  # Biais des neurones cachés (H,)

    activation = X@W + b  # Produit matriciel + biais cachés
    H = sigmoid(activation)  # Application de la fonction sigmoïde

    return H


def sortie_entree_RBM(RBM, H):


    W = RBM["W"]  # Matrice des poids (D, H)
    a = RBM["a"]  # Biais des neurones visibles (D,)

    activation = np.dot(H, W.T) + a  # Produit matriciel + biais visibles
    X_reconstruit = sigmoid(activation)  # Application de la fonction sigmoïde

    return X_reconstruit


def train_RBM(dataset, n_hidden, epochs, learning_rate, batch_size):
    """
    Entraîne un RBM en utilisant l'algorithme de Contrastive Divergence-1 (CD-1).
    But : Apprendre une représentation efficace des images
    Méthode : Réduire la différence entre la phase positive (entrée -> sorties) et la phase negative (entrée-> sorties -> entrées -> sorties).
    """
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_visible = next(iter(train_loader))[0].shape[1]  # nbr_image_class, nbr neurones visibles = dim image
    RBM = init_RBM(n_visible, n_hidden)  # Initialisation du RBM

    for epoch in range(epochs):

        # Parcourir les mini-batches
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.numpy(), Y_batch.numpy()
            # étape positive : calcul des probabilités des neurones cachés
            H_pos = entree_sortie_RBM(RBM, X_batch)

            # olutput proba des hidden units (0,1), pour chaque images (donc de shape (batch_size,n_hidden))
            #pour travailler avec des bianires plutot que des proba
            H_sample = (np.random.rand(*H_pos.shape) < H_pos).astype(np.float32)
            # étape négative : reconstruction des visibles à partir des cachés
            X_neg = sortie_entree_RBM(RBM, H_sample)

            # calcul des probabilités des neurones cachés pour les données reconstruites
            H_neg = entree_sortie_RBM(RBM, X_neg)

            # mise à jour des poids et des biais
            grad_W = (np.dot(X_batch.T, H_pos) - np.dot(X_neg.T, H_neg))/batch_size
            grad_a = (np.sum(X_batch - X_neg, axis=0)) / batch_size
            grad_b = (np.sum(H_pos - H_neg, axis=0)) / batch_size

            RBM["W"] += learning_rate * grad_W
            RBM["a"] += learning_rate * grad_a
            RBM["b"] += learning_rate * grad_b
        # Calcul de la MSE à la fin de chaque époch
        X_reconstruit = sortie_entree_RBM(RBM, entree_sortie_RBM(RBM, dataset.get_all()[0])) #on passe tout notre dataset X en entrée
        mse = np.mean((dataset.get_all()[0] - X_reconstruit) ** 2)

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse:.8f}")

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
        plt.suptitle(f"Images générées après {n_iterations} itérations de Gibbs pour la classe {35} ")
        plt.show()

    return generated_images



#Définition du custom pytorch dataset pour aussi utiliser pytorch avec les données AlphaDigits
class binaryalphadigs_dataset(Dataset):
    def __init__(self, data, indices_classes):

        self.X,self.Y = [],[]
        images = data['dat']

        for idx in indices_classes:
            for img in images[idx]:  # Parcourir les 39 images de la classe choisie
                self.X.append(img.flatten())  # Transformer l'image en vecteur (1D)
                self.Y.append(idx)
        self.X, self.Y = np.array(self.X), np.array(self.Y)

    def __len__(self):
        return len(self.X) #shape X = (len(indices_classes)*39, dim_images[0]*dim_images[1] = 320)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        return x_sample, y_sample

    def shape(self):
        return self.X.shape

    def get_all(self):
        "pour obtenir tout le dataset"
        return self.X,self.Y

    def update(self,X_new):
        "utile notamment pour le passage d'information à travers le couche du DBN sans à avoir à créer un autre dataset"
        self.X = X_new


class CustomMNISTDataset(Dataset):
    def __init__(self, limit, root="datasets", train=True, download=False, transform=None):
        # Charger le dataset MNIST
        self.mnist_dataset = MNIST(root=root, train=train, download=download, transform=transform)

        # Initialiser les listes pour stocker les données et les labels
        self.X = []
        self.Y = []

        # Remplir les listes avec les données de MNIST
        for img, label in self.mnist_dataset:
            self.X.append(img.numpy().flatten())  # Transformer l'image en vecteur (1D)
            self.Y.append(label)
            if len(self.X) >= limit : break

        # Convertir les listes en tableaux numpy
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def __len__(self):
        return len(self.X)  # Retourne la taille du dataset

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        return x_sample, y_sample

    def shape(self):
        return self.X.shape  # Retourne la forme des données

    def get_all(self):
        """Pour obtenir tout le dataset"""
        return self.X, self.Y

    def update(self, X_new):
        """Mettre à jour les données"""
        self.X = X_new


