# Présentation du Projet

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

Le projet vise à comparer les performances d'un modèle pré-entraîné puis fine-tuné pour la classification à celles d'un modèle entraîné uniquement pour la tâche de classification, sans pré-entraînement.


##  Installation et Utilisation

Fichier main.py
