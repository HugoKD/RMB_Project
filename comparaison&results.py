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
        'learning_rate_GD_DNN': 0.1,
        'learning_rate_RBM': 0.1,
        'batch_size_GD_DNN': 32,
        'batch_size_RBM': 32,
        'n_donnees': 3000,
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
        'learning_rate_GD_DNN': 0.1,
        'learning_rate_RBM': 0.1,
        'batch_size_GD_DNN': 32,
        'batch_size_RBM': 32,
        'n_donnees': 3000,
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
        'learning_rate_GD_DNN': 0.1,
        'learning_rate_RBM': 0.1,
        'batch_size_GD_DNN': 32,
        'batch_size_RBM': 32,
        'n_donnees': 3000,
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

from itertools import product
import torch
from torchvision import transforms
import numpy as np

from itertools import product
import torch
from torchvision import transforms

def grid_search(params_grid, n_donnees):
    '''
    Perform a grid search over the provided parameter combinations to find optimal parameters for model performance.

    Parameters:
    - params_grid: Dictionary with parameter names as keys and lists of values to try.

    Returns:
    - best_result: Tuple with best parameter combination and corresponding error rate.
    '''

    results = {}
    best_config = None
    best_error_rate = float('inf')

    for epochs, num_layers, num_units, lr, batch_size in product(
        params_grid['epochs'],
        params_grid['num_layers'],
        params_grid['num_units_per_layer'],
        params_grid['lr'],
        params_grid['batch_size']):

        print(f"Testing combination: epochs_RBM={epochs}, layers={num_layers}, units={num_units}, lr_RBM={lr}, batch_size={batch_size}")

        parametres = {
            'n_epochs_GD_DNN': epochs,
            'n_epochs_RBM': epochs,
            'learning_rate_GD_DNN': lr,
            'learning_rate_RBM': lr,
            'batch_size_GD_DNN': batch_size,
            'batch_size_RBM': batch_size,
            'n_donnees': n_donnees,
            'n_classes': 10,
            'image_size': (20, 16),
            "indices_classes_to_trained_on": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }

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

        parametres['taille_reseau'] = [parametres["image_size"][0] * parametres["image_size"][1]] + [num_units] * num_layers

        dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform, limit=parametres['n_donnees'])
        DNN = init_DNN(parametres['taille_reseau'], nbr_classes=parametres['n_classes'])
        pt_model = pretrain_DNN(DNN=DNN, sizes = parametres["taille_reseau"], learning_rate=parametres["learning_rate_RBM"],dataset=dataset, epochs=parametres["n_epochs_RBM"],
                                batch_size=parametres["batch_size_RBM"], image_size=parametres['image_size'], plot = False)

        dataset = CustomMNISTDataset(root="datasets", train=True, download=False, transform=transform, limit=parametres['n_donnees'])
        trained_DNN = retropropagation(pt_model, dataset, parametres['n_epochs_GD_DNN'], parametres['learning_rate_GD_DNN'],
                                       parametres['batch_size_GD_DNN'], parametres['image_size'],
                                       n_classes=parametres['n_classes'], plot=False)

        test_dataset = CustomMNISTDataset(root="datasets", train=False, download=False, transform=transform, limit=parametres['n_donnees'])
        true_labels = test_dataset.Y

        error_rate = DNN_test(dataset=test_dataset, DNN=trained_DNN, true_labels=true_labels)

        key = f"epochs_RBM={epochs}_layers={num_layers}_units={num_units}_lr_RBM={lr}_batch_size={batch_size}"
        results[key] = error_rate

        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_config = key

        print(f"Error rate: {error_rate}\n")

    print(f"\nBest configuration: {best_config} with error rate: {best_error_rate}\n")
    return best_config, best_error_rate, results



#compare_number_layers([1,2,3,4,5], units_per_layer = 200) #test (train deja fait)
#compare_number_units_per_layer([100,200,700],num_layers = 2) #train 'on refait en rouge'
#compare_number_of_data([100,1000,2000,2000,3000,4000,5000,6000,7000], units = 200, num_layers = 2) #train & test
# Example parameter grid for grid search

params_grid = {
    'epochs': [50,500,1000,1500,2000,2500],
    'num_layers': [2,4,5,8],
    'num_units_per_layer': [200, 500,850],
    'lr': [0.5,0.1,0.05],
    'batch_size': [32, 64],
}


n_donnees = 4000

best_config, best_error_rate, results = grid_search(params_grid=params_grid, n_donnees=n_donnees)

print(f"Best Configuration: {best_config}")
print(f"Best Error Rate: {best_error_rate}")

# Display all results
print("\nAll Results:")
for config, error in results.items():
    print(f"{config}: {error}")
