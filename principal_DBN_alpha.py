from principal_RBM_aplha import *



def init_DBN(sizes):
    """
    Initialise un DBN en empilant plusieurs RBMs."""
    DBN = []
    for i in range(len(sizes) - 1):
        n_visible = sizes[i]      # Nombre de neurones visibles (couche actuelle)
        n_hidden = sizes[i + 1]   # Nombre de neurones cachés (couche suivante)

        rbm = init_RBM(n_visible, n_hidden)

        DBN.append(rbm)

    return DBN



def train_DBN(dataset, sizes, epochs, learning_rate, batch_size, image_size, plot = False):
    """
    Entraîne un DBN de manière non supervisée en utilisant la procédure Greedy Layer-Wise Training.
    """
    assert image_size[0] * image_size[1] == sizes[0], "La première couche doit correspondre à la taille flatten de l'image"
    DBN = init_DBN(sizes)

    # Données d'entrée pour la première couche
    input_data = dataset
    # Entraînement couche par couche
    for i in range(len(DBN)):
        print(f"Entraînement de la couche {i + 1}/{len(DBN)}...")

        # Entraînement du RBM actuel
        DBN[i] = train_RBM(
            dataset,
            n_hidden=sizes[i + 1],
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            plot = plot
        )

        # Propagation des données à travers le RBM entraîné
        if i < len(DBN) - 1:  # Ne pas propager pour la dernière couche

            X_output = entree_sortie_RBM(RBM=DBN[i], X=input_data.get_all()[0])
            X_sample = (np.random.rand(*X_output.shape) < X_output).astype(np.float32) #apporter de la stochasticité
            dataset.update(X_sample)


    print("Pré-entraînement du DBN terminé !!")
    return DBN


def generer_image_DBN(DBN, n_iterations=200, n_images=10, image_shape=(20, 16), plot = True):

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
        plt.suptitle(f"Images générées par le DBN après {n_iterations} itérations de Gibbs")
        plt.show()

    return generated_images

