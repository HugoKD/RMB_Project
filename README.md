# 📌 Présentation du Projet

Ce petit projet explore et manipule des modèles d’apprentissage automatique sur deux jeux de données : **MNIST** et **Binary AlphaDigits**.  
L'objectif principal est d'évaluer et de comparer les performances de trois architectures de réseaux de neurones spécifiques :

### 🔹 Machines de Boltzmann Restreintes (RBM)
- Modèles probabilistes non supervisés pour apprendre des représentations latentes.
- Composées d'une couche visible et d'une couche cachée, sans connexions internes.
- Utilisation de l'algorithme **Contrastive Divergence** pour l'apprentissage.

### 🔹 Deep Belief Networks (DBN)
- Empilement hiérarchique de plusieurs RBM pour une pré-initialisation efficace des poids.
- Permet l’extraction de caractéristiques hiérarchiques des données.

### 🔹 Deep Neural Networks (DNN)
- Réseaux de neurones profonds entièrement connectés avec **rétropropagation**.
- Entraînement **bout à bout** pour une optimisation directe sur tout le réseau.

##  Méthodologie

Le projet vise à **comparer ces architectures** en termes de :

-  **Capacité de représentation** des données.
-  **Performances en classification** sur MNIST et Binary AlphaDigits.
-  **Pouvoir de généralisation** et **robustesse** face aux variations des données (bruit, transformations).
-  **Temps d'apprentissage** et **analyse des performances** (précision, matrice de confusion).

## 📂 Données

Les jeux de données utilisés proviennent de **Kaggle**

