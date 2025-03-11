from principal_DNN_MNIST import *


def main(generate, evaluate, parametres, model, dataset_choice='MNIST', pretrain=True):
    """
    Fonction principale pour organiser le projet.

    Args:
        generate (bool): Si l'on veut utiliser les modèles pour générer des images (uniquement pour DBN et RBM).
        evaluate (bool): Si l'on veut évaluer le modèle DNN.
        parametres (dict): Les paramètres du modèle.
        model (str): Le type de modèle à utiliser ('RBM', 'DBN', 'DNN').
        dataset_choice (str): Choix du dataset ('MNIST' ou 'AlphaDigits').
        pretrain (bool): Indique si le DNN doit être pré-entraîné ou non.

    Returns:
        None
    """

    # Définition de la transformation PyTorch
    class BinarizeImage:
        def __init__(self, threshold=128):
            self.threshold = threshold

        def __call__(self, x):
            return torch.where(x > self.threshold / 255.0, torch.tensor(1.0), torch.tensor(0.0))

    transform = transforms.Compose([
        transforms.ToTensor(),
        BinarizeImage(threshold=128),
        transforms.Resize((parametres["image_size"][0], parametres["image_size"][1])),
    ])

    # Sélection du dataset
    if dataset_choice == 'AlphaDigits':
        file_path = os.path.join("datasets", "binaryalphadigs.mat")
        data = scipy.io.loadmat(file_path)
        dataset = binaryalphadigs_dataset(data=data, indices_classes=parametres.get("indices_classes_to_trained_on", [0, 1, 2, 3, 4, 5]))
    else:
        dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform, limit=parametres["n_données"])

    # Gestion du modèle
    if model == "RBM":
        trained_rbm = train_RBM(dataset=dataset, n_hidden=parametres["n_hidden"],
                                epochs=parametres["epochs"], learning_rate=parametres["learning_rate"],
                                batch_size=parametres["batch_size"])
        if generate:
            generer_image_RBM(trained_rbm, n_iterations=1000, n_images=10, image_shape=(20, 16), plot=True)

    elif model == "DBN":
        DBN_trained = train_DBN(dataset, parametres['taille_reseau'], epochs=parametres["epochs"],
                                learning_rate=parametres["learning_rate"], batch_size=parametres["batch_size"],
                                image_size=parametres["image_size"])
        if generate:
            generer_image_DBN(DBN_trained, n_iterations=1000, n_images=10, image_shape=(20, 16), plot=True)

    elif model == "DNN":
        DNN = init_DNN(parametres["taille_reseau"], nbr_classes=parametres["n_classes"])

        if pretrain:
            DNN = pretrain_DNN(DNN=DNN, dataset=dataset, epochs=parametres["n_epochs_RBM"],
                               learning_rate=parametres["learning_rate_RBM"], batch_size=parametres["batch_size_RBM"],
                               image_size=parametres["image_size"], sizes=parametres['taille_reseau'], plot=True)
            print("Pre-training done...")
            if dataset_choice == 'AlphaDigits':
                file_path = os.path.join("datasets", "binaryalphadigs.mat")
                data = scipy.io.loadmat(file_path)
                dataset = binaryalphadigs_dataset(data=data,
                                                  indices_classes=parametres.get("indices_classes_to_trained_on",
                                                                                 [0, 1, 2, 3, 4, 5]))
            else:
                dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                             limit=parametres["n_données"])

        DNN = retropropagation(DNN, dataset, parametres["n_epochs_GD_DNN"], parametres["learning_rate_GD_DNN"],
                               parametres["batch_size_GD_DNN"], parametres["image_size"],
                               n_classes=parametres["n_classes"], plot=True)
        print('Fine-tuning done...')

        if evaluate:
            if dataset_choice == 'AlphaDigits':
                file_path = os.path.join("datasets", "binaryalphadigs.mat")
                data = scipy.io.loadmat(file_path)
                test_dataset = binaryalphadigs_dataset(data=data,
                                                  indices_classes=parametres.get("indices_classes_to_trained_on",
                                                                                 [0, 1, 2, 3, 4, 5]))
            else:
                test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform,
                                                  limit=parametres["n_données"])

            true_labels = test_dataset.Y
            error_rate = DNN_test(DNN=DNN, dataset=test_dataset, true_labels=true_labels)
            print(f"Error Rate: {error_rate}")

    else:
        print("Invalid model type. Choose 'RBM', 'DBN', or 'DNN'.")


# Example usage
parametres = {
    'n_hidden': 10,
    'epochs': 10,
    'learning_rate': 0.1,
    'batch_size': 82,
    'image_size': (20, 16),
    'n_données': 3000,
    'n_classes': 10,
    'taille_reseau': [320, 784, 500, 200],
    'n_epochs_GD_DNN': 350,
    'n_epochs_RBM': 350,
    'learning_rate_GD_DNN': 0.1,
    'learning_rate_RBM': 0.1,
    'batch_size_GD_DNN': 124,
    'batch_size_RBM': 124,
    'indices_classes_to_trained_on': [0, 1, 2, 3, 4, 5]
}


main(generate=True, evaluate=True, parametres=parametres, model="DNN", dataset_choice='MNIST', pretrain=True)
