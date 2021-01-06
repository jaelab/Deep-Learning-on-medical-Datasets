# INF8225-Project
## Exploration de diverses méthodes de segmentation d'images médicales.
### Plusieurs ensembles de données sont utilisés soient BraTS-18, ISIC-2017, DRIVE et CHAOS.
### Les modèles explorés sont un encodeur automatique régularisé, un SegAN, une dérivée de U-Net (BCDU-Net) et un modèle multi-scale self-guided. Chacun de ces modèles a un fichier main personnalisé.

### Pour BCDU-Net:
Implémentation et pré-traitement inspiré de https://github.com/rezazad68/BCDU-Net. Les données peuvent être obtenue à travers le lien suivant: https://drive.google.com/file/d/17wVfELqgwbp4Q02GD247jJyjq6lwB0l6/view. Assurez vous que le fichier soit dans le répertoire rawdata\ et que celui-ci se nomme DRIVE. Voici les répertoires qu'il faut créer pour rouler le code. 
Il faut créer les répertoires "BCDU_models", "Preprocessed_Images" et "Tests" dans le répertoire "BCDU-net" pour avoir l'ordre suivant:
architectures/BCDU_net/Preprocessed_Images/... Aussi il faut ajouter les répertoires "Patches" et "CLAHE" dans "Preprocessed_Images".

Pour démarrer l'entraînement, allez dans le fichier "INF8225-Project" et ouvrez une console à partir de cette destination. Tapez la commande:
```
python mains/train_bcdunet.py --train --eval
```
pour démarrer l'entraînement et effectuer l'évaluation du modèle par la suite. 

### Pour Autoencoder 3D:
Il est possible de downloader les données d'entrainment sur le lien: http://academictorrents.com/details/a9e2741587d42ef6139aa474a95858a17952b3a5. Ensuite extraire les données dans le dossier rawdata/.

Pour démarrer l'entraînement, allez dans le fichier "INF8225-Project" et ouvrez une console à partir de cette destination. Tapez la commande:
```
python mains/train_nvdlmed.py --create_hierarchy --train --eval
```
pour démarrer l'entraînement et effectuer l'évaluation du modèle par la suite (La commande --create_hierarchy va créer les dossiers nécessaires pour l'utilisation du modèle).

### Pour MS-Dual-Guided:
Il est possible de downloader les données d'entrainment sur le lien: https://zenodo.org/record/3431873#.XqTWaGhKiUk. Ensuite extraire les données dans le dossier rawdata/.

Pour démarrer l'entraînement, allez dans le fichier "INF8225-Project" et ouvrez une console à partir de cette destination. Tapez la commande:
```
python mains/train_danet.py --create_hierarchy --train --eval
```
pour démarrer l'entraînement et effectuer l'évaluation du modèle par la suite (La commande --create_hierarchy va créer les dossiers nécessaires pour l'utilisation du modèle).

### Pour SegAN: 
Il est possible de downloader les données d'entraînement, de validation et de test aux liens suivants:
 * entraînement: https://challenge.kitware.com/#phase/5841916ccad3a51cc66c8db0
 * validation: https://challenge.kitware.com/#phase/584b08eecad3a51cc66c8e1f
 * test: https://challenge.kitware.com/#phase/584b0afacad3a51cc66c8e24

Ensuite, extraire les données et les mettre dans un dossier nommé datasets/data/.

Pour démarrer l'entraînement, allez dans le fichier "INF8225-Project" et ouvrez une console à partir de cette destination. Tapez la commande:
```
python mains/train_segan.py
```
pour démarrer l'entraînement et effectuer l'évaluation du modèle par la suite. 



