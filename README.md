Ce projet est un détecteur de visage réalisé dans le cadre de l'UV SY32 de l'UTC.

Important : pour faire fonctionner les scripts, il est nécessaire d'installer opencv et imutils voici les commandes à entrée dans une console python : 

pip install opencv-python
pip install imutils


Ce dossier contient 4 fichiers : 

- train.py : qui est le script d'entrainement de mon détecteur de visage, ce fichier génère un modèle de classifier : classifier_model.sav qui est utilisé par test.py 

- test.py : qui est le fichier permettant de tester le detecteur de visage. Ce fichier génère un fichier detection.txt qui est le résultat des détections

Tutoriel d'utilisation :

-Placer le fichier project_train contenant les images d'entrainement et le fichier label_train dans le répertoire de ce projet.
-Placer le fichier test contenant les images dans le répertoire de ce fichier.

- Lancer le script train.py pour générer le modèle de classifier (optionnel, celui-ci est déjà dans le repertoire), attention, la génération peut prendre beaucoup de temps selon votre config !

- Lancer le script test.py pour générer le fichier detection.txt

- (optionnel) Si vous souhaitez voir l'algorithme travailler sur une image visuellement, il est possible de supprimer des commentaires dans le script test.py qui offrent cette fonctionnalité.





