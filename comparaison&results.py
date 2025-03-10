from principal_DNN_MNIST import *

###########################################################################################################################################
################################################################5.2 Analyse################################################################
###########################################################################################################################################



def compare_number_layers(layers,units_per_layer,plot=True):
    '''
    Compare two models (one pre-trained and the other initialized randomly) in terms of error rate
    across different numbers of layers on MNIST.

    Returns the error rates of each model for each number of layers as a dictionary.

    Parameters:
    - layers: List of different numbers of layers to test.
    - plot: Boolean to indicate if the results should be plotted.
    '''


    error_rates_ft_model = {}
    error_rates_non_pt_model = {}

    parametres = {
        'n_epochs_GD_DNN': 450,
        'n_epochs_RBM': 200,
        'learning_rate_GD_DNN': 0.001,
        'learning_rate_RBM': 0.001,
        'batch_size_GD_DNN': 32,
        'batch_size_RBM': 32,
        'n_donnees': 4000,
        'n_classes': 10,
        'image_size': (20, 16),  # Convert MNIST images to 20x16 instead of 28x28
        "indices_classes_to_trained_on": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    class BinarizeImage:
        def __init__(self, threshold=128):
            self.threshold = threshold

        def __call__(self, x):
            # Binarisation de l'image : pixels supérieurs au seuil deviennent 1, les autres deviennent 0
            return torch.where(x > self.threshold / 255.0, torch.tensor(1.0), torch.tensor(0.0))

    # Préprocessing
    transform = transforms.Compose([  # Pour MNIST (utilisation de pytorch
        ToTensor(),
        BinarizeImage(threshold=128),  # binariser blanc/noir
        transforms.Resize((parametres["image_size"][0], parametres["image_size"][1])),
    ])

    for i, num_layers in enumerate(layers):
        print(f"{num_layers} nombre de couches, en cours ........\n")
        parametres['taille_reseau'] = [parametres["image_size"][0] * parametres["image_size"][1]] + [units_per_layer] * num_layers
        print('taille_reseau', parametres["taille_reseau"])



        # Pre-training and fine-tuning
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                          limit=parametres["n_donnees"])
        DNN = init_DNN(parametres["taille_reseau"], nbr_classes=parametres["n_classes"])
        pre_trained_DNN = pretrain_DNN(DNN=DNN, dataset=train_dataset, epochs=parametres["n_epochs_RBM"],
                                       learning_rate=parametres["learning_rate_RBM"],
                                       batch_size=parametres["batch_size_RBM"],
                                       image_size=parametres["image_size"], sizes=parametres['taille_reseau'],
                                       plot=False)
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                          limit=parametres["n_donnees"])

        ft_modele = retropropagation(pre_trained_DNN, train_dataset, parametres["n_epochs_GD_DNN"],
                                     parametres["learning_rate_GD_DNN"],
                                     parametres["batch_size_GD_DNN"], parametres["image_size"],
                                     n_classes=parametres["n_classes"], plot=True)
        ## Même chose mais sans pré-entrainement
        DNN = init_DNN(parametres["taille_reseau"], nbr_classes=parametres["n_classes"])
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                           limit=parametres["n_donnees"])
        non_pt_modele = retropropagation(DNN=DNN, dataset = train_dataset,epochs= parametres["n_epochs_GD_DNN"],
                                     learning_rate= parametres["learning_rate_GD_DNN"],
                                     batch_size= parametres["batch_size_GD_DNN"], image_size= parametres["image_size"],
                                     n_classes=parametres["n_classes"], plot=True)

        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform,
                                          limit=parametres["n_donnees"])
        true_labels = test_dataset.Y
        error_r_ft_modele = DNN_test(dataset=test_dataset, DNN=ft_modele, true_labels=true_labels)
        #Même chose pour le modèle sans pré-entrainement
        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform,
                                          limit=parametres["n_donnees"])
        true_labels = test_dataset.Y
        error_r_non_pt_modele = DNN_test(dataset=test_dataset, DNN=non_pt_modele, true_labels=true_labels)
        print(f"Error rate for non finetuned model :  {error_r_ft_modele} \n")

        error_rates_ft_model[num_layers] = error_r_ft_modele
        error_rates_non_pt_model[num_layers] = error_r_non_pt_modele

    if plot:
        layers_str = list(error_rates_ft_model.keys())
        ft_errors = list(error_rates_ft_model.values())
        non_pt_errors = list(error_rates_non_pt_model.values())
        plt.figure(figsize=(10, 6))
        plt.plot(layers_str, ft_errors, label='Pre-trained Model', marker='o', color='red')
        plt.plot(layers_str, non_pt_errors, label='Non Pre-trained Model', marker='o', color='blue')
        plt.xlabel('Number of Layers')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Number of Layers')
        plt.legend()
        plt.grid(True)
        plt.show()

    return


def compare_number_units_per_layer(units,num_layers, plot=True):
    '''
    Compare two models (one pre-trained and the other initialized randomly) in terms of error rate
    across different numbers of layers on MNIST.

    Returns the error rates of each model for each number of layers as a dictionary.

    Parameters:
    - layers: List of different numbers of layers to test.
    - plot: Boolean to indicate if the results should be plotted.
    '''

    error_rates_ft_model = {}
    error_rates_non_pt_model = {}
    parametres = {
        'n_epochs_GD_DNN': 300,
        'n_epochs_RBM': 150,
        'learning_rate_GD_DNN': 0.001,
        'learning_rate_RBM': 0.001,
        'batch_size_GD_DNN': 32,
        'batch_size_RBM': 32,
        'n_donnees': 5000,
        'n_classes': 10,
        'image_size': (20, 16),  # Convert MNIST images to 20x16 instead of 28x28
        "indices_classes_to_trained_on": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    class BinarizeImage:
        def __init__(self, threshold=128):
            self.threshold = threshold

        def __call__(self, x):
            # Binarisation de l'image : pixels supérieurs au seuil deviennent 1, les autres deviennent 0
            return torch.where(x > self.threshold / 255.0, torch.tensor(1.0), torch.tensor(0.0))

    # Préprocessing
    transform = transforms.Compose([  # Pour MNIST (utilisation de pytorch
        ToTensor(),
        BinarizeImage(threshold=128),  # binariser blanc/noir
        transforms.Resize((parametres["image_size"][0], parametres["image_size"][1])),
    ])



    for i, num_units in enumerate(units):
        print(f"{num_units} nombre de units, en cours ........\n")
        parametres['taille_reseau'] = [parametres["image_size"][0] * parametres["image_size"][1]] + [num_units] * num_layers
        print('taille_reseau', parametres["taille_reseau"])

        # Pre-training and fine-tuning
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                           limit=parametres["n_donnees"])
        DNN = init_DNN(parametres["taille_reseau"], nbr_classes=parametres["n_classes"])
        pre_trained_DNN = pretrain_DNN(DNN=DNN, dataset=train_dataset, epochs=parametres["n_epochs_RBM"],
                                       learning_rate=parametres["learning_rate_RBM"],
                                       batch_size=parametres["batch_size_RBM"],
                                       image_size=parametres["image_size"], sizes=parametres['taille_reseau'],
                                       plot=False)
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                           limit=parametres["n_donnees"])

        ft_modele = retropropagation(pre_trained_DNN, train_dataset, parametres["n_epochs_GD_DNN"],
                                     parametres["learning_rate_GD_DNN"],
                                     parametres["batch_size_GD_DNN"], parametres["image_size"],
                                     n_classes=parametres["n_classes"], plot=True)
        ## Même chose mais sans pré-entrainement
        DNN = init_DNN(parametres["taille_reseau"], nbr_classes=parametres["n_classes"])
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                           limit=parametres["n_donnees"])
        non_pt_modele = retropropagation(DNN=DNN, dataset=train_dataset, epochs=parametres["n_epochs_GD_DNN"],
                                         learning_rate=parametres["learning_rate_GD_DNN"],
                                         batch_size=parametres["batch_size_GD_DNN"],
                                         image_size=parametres["image_size"],
                                         n_classes=parametres["n_classes"], plot=True)

        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform,
                                          limit=parametres["n_donnees"])
        true_labels = test_dataset.Y
        error_r_ft_modele = DNN_test(dataset=test_dataset, DNN=ft_modele, true_labels=true_labels)
        # Même chose pour le modèle sans pré-entrainement
        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform,
                                          limit=parametres["n_donnees"])
        true_labels = test_dataset.Y
        error_r_non_pt_modele = DNN_test(dataset=test_dataset, DNN=non_pt_modele, true_labels=true_labels)
        print(f"Error rate for non finetuned model :  {error_r_ft_modele} \n")

        error_rates_ft_model[num_units] = error_r_ft_modele
        error_rates_non_pt_model[num_units] = error_r_non_pt_modele

    if plot:
        units_str = list(error_rates_ft_model.keys())
        ft_errors = list(error_rates_ft_model.values())
        non_pt_errors = list(error_rates_non_pt_model.values())
        plt.figure(figsize=(10, 6))
        plt.plot(units_str, ft_errors, label='Pre-trained Model', marker='o', color='red')
        plt.plot(units_str, non_pt_errors, label='Non Pre-trained Model', marker='o', color='blue')
        plt.xlabel('Number of units per layer')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Number of units per layer')
        plt.legend()
        plt.grid(True)
        plt.show()

def compare_number_of_data(datas, units, num_layers, plot=True):
    '''
    Compare two models (one pre-trained and the other initialized randomly) in terms of error rate
    across different numbers of layers on MNIST.

    Returns the error rates of each model for each number of layers as a dictionary.

    Parameters:
    - layers: List of different numbers of layers to test.
    - plot: Boolean to indicate if the results should be plotted.
    '''

    error_rates_ft_model = {}
    error_rates_non_pt_model = {}

    parametres = {
        'n_epochs_GD_DNN': 10,
        'n_epochs_RBM': 10,
        'learning_rate_GD_DNN': 0.001,
        'learning_rate_RBM': 0.001,
        'batch_size_GD_DNN': 32,
        'batch_size_RBM': 32,
        'n_donnees': 4000,
        'n_classes': 10,
        'image_size': (20, 16),  # Convert MNIST images to 20x16 instead of 28x28
        "indices_classes_to_trained_on": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    parametres['taille_reseau'] = [parametres["image_size"][0] * parametres["image_size"][1]] + [units] * num_layers

    class BinarizeImage:
        def __init__(self, threshold=128):
            self.threshold = threshold

        def __call__(self, x):
            # Binarisation de l'image : pixels supérieurs au seuil deviennent 1, les autres deviennent 0
            return torch.where(x > self.threshold / 255.0, torch.tensor(1.0), torch.tensor(0.0))

    # Préprocessing
    transform = transforms.Compose([  # Pour MNIST (utilisation de pytorch
        ToTensor(),
        BinarizeImage(threshold=128),  # binariser blanc/noir
        transforms.Resize((parametres["image_size"][0], parametres["image_size"][1])),
    ])

    for i, data in enumerate(datas):
        print(f"{data} nombre de units, en cours ........\n")
        parametres['n_donnees'] = datas[i]
        print('data', datas[i])

        # Pre-training and fine-tuning
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                           limit=parametres["n_donnees"])
        DNN = init_DNN(parametres["taille_reseau"], nbr_classes=parametres["n_classes"])
        pre_trained_DNN = pretrain_DNN(DNN=DNN, dataset=train_dataset, epochs=parametres["n_epochs_RBM"],
                                       learning_rate=parametres["learning_rate_RBM"],
                                       batch_size=parametres["batch_size_RBM"],
                                       image_size=parametres["image_size"], sizes=parametres['taille_reseau'],
                                       plot=False)
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                           limit=parametres["n_donnees"])

        ft_modele = retropropagation(pre_trained_DNN, train_dataset, parametres["n_epochs_GD_DNN"],
                                     parametres["learning_rate_GD_DNN"],
                                     parametres["batch_size_GD_DNN"], parametres["image_size"],
                                     n_classes=parametres["n_classes"], plot=True)
        ## Même chose mais sans pré-entrainement
        DNN = init_DNN(parametres["taille_reseau"], nbr_classes=parametres["n_classes"])
        train_dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform,
                                           limit=parametres["n_donnees"])
        non_pt_modele = retropropagation(DNN=DNN, dataset=train_dataset, epochs=parametres["n_epochs_GD_DNN"],
                                         learning_rate=parametres["learning_rate_GD_DNN"],
                                         batch_size=parametres["batch_size_GD_DNN"],
                                         image_size=parametres["image_size"],
                                         n_classes=parametres["n_classes"], plot=True)

        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform,
                                          limit=parametres["n_donnees"])
        true_labels = test_dataset.Y
        error_r_ft_modele = DNN_test(dataset=test_dataset, DNN=ft_modele, true_labels=true_labels)
        # Même chose pour le modèle sans pré-entrainement
        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform,
                                          limit=parametres["n_donnees"])
        true_labels = test_dataset.Y
        error_r_non_pt_modele = DNN_test(dataset=test_dataset, DNN=non_pt_modele, true_labels=true_labels)
        print(f"Error rate for non finetuned model :  {error_r_ft_modele} \n")

        error_rates_ft_model[data] = error_r_ft_modele
        error_rates_non_pt_model[data] = error_r_non_pt_modele

    if plot:
        data_str = list(error_rates_ft_model.keys())
        ft_errors = list(error_rates_ft_model.values())
        non_pt_errors = list(error_rates_non_pt_model.values())
        plt.figure(figsize=(10, 6))
        plt.plot(data_str, ft_errors, label='Pre-trained Model', marker='o', color='red')
        plt.plot(data_str, non_pt_errors, label='Non Pre-trained Model', marker='o', color='blue')
        plt.xlabel('Number of data used for training')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Number of  data used for training')
        plt.legend()
        plt.grid(True)
        plt.show()
    return

import matplotlib.pyplot as plt
from itertools import product

def grid_search(params_grid):
    '''
    Perform a grid search over the provided parameter combinations to find optimal parameters for model performance.

    Parameters:
    - params_grid: Dictionary with parameter names as keys and lists of values to try.

    Returns:
    - results: Dictionary of parameter combinations and corresponding error rates.
    '''



    results = {}

    for num_data, num_layers, num_units, lr in product(params_grid['num_data'],
                                                       params_grid['num_layer'],
                                                       params_grid['num_units_per_layer'],
                                                       params_grid['lr']):

        print(f"Testing combination: data={num_data}, layers={num_layers}, units={num_units}, lr={lr}")

        parametres = {
            'n_epochs_GD_DNN': 50,
            'n_epochs_RBM': 20,
            'learning_rate_GD_DNN': lr,
            'learning_rate_RBM': lr,
            'batch_size_GD_DNN': 32,
            'batch_size_RBM': 32,
            'n_donnees': num_data,
            'n_classes': 10,
            'image_size': (20, 16),
            "indices_classes_to_trained_on": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }

        class BinarizeImage:
            def __init__(self, threshold=128):
                self.threshold = threshold

            def __call__(self, x):
                # Binarisation de l'image : pixels supérieurs au seuil deviennent 1, les autres deviennent 0
                return torch.where(x > self.threshold / 255.0, torch.tensor(1.0), torch.tensor(0.0))

        # Préprocessing
        transform = transforms.Compose([  # Pour MNIST (utilisation de pytorch
            ToTensor(),
            BinarizeImage(threshold=128),  # binariser blanc/noir
            transforms.Resize((parametres["image_size"][0], parametres["image_size"][1])),
        ])

        parametres['taille_reseau'] = [parametres["image_size"][0] * parametres["image_size"][1]] + [num_units] * num_layers

        dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform, limit=num_data)
        DNN = init_DNN(parametres['taille_reseau'], nbr_classes=parametres['n_classes'])
        trained_DNN = retropropagation(DNN, dataset, parametres['n_epochs_GD_DNN'], lr,
                                       parametres['batch_size_GD_DNN'], parametres['image_size'],
                                       n_classes=parametres['n_classes'], plot=False)

        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform, limit=num_data)
        true_labels = test_dataset.Y
        error_rate = DNN_test(dataset=test_dataset, DNN=trained_DNN, true_labels=true_labels)

        key = f"data={num_data}_layers={num_layers}_units={num_units}_lr={lr}"
        results[key] = error_rate
        print(f"Error rate: {error_rate}\n")

    return results


compare_number_layers([1,2,3,4,5], units_per_layer = 2) #test (train deja fait)
compare_number_units_per_layer([100,200,700],num_layers = 2) #train 'on refait en rouge'
compare_number_of_data([100,1000,2000,2000,3000,4000,5000,6000,7000], units = 200, num_layers = 2) #train & test

