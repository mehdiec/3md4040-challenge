# Commandes à lancer pour utiliser le main.py

Clonage du repo git

   ```sh 
   git clone https://gitlab-student.centralesupelec.fr/hamza.benslimane/3md4040-challenge.git
   cd 3md4040-challenge
   ```
Lancer le train avec main.py

   ```sh 
   python3 src/main.py train PATH_TO_TRAINING_SET
   

   ```
   Un dossier logs sera alors crée dans le répertoire. Celui-ci contiendra le best_model.pt ainsi que qu'un récapitulatif du train et score.

Lancer le test avec main.py (avec notre meilleur modèle)

   ```sh 
   python3 src/main.py test resnet34_2.2/best_model.pt PATH_TO_TEST_SET
   ```

Vous trouverez le fichier csv (résultat du test) dans le dossier Results

# Commandes utilisables pour changer les paramètres de train et test 

Entrainer un modèle en normalisant les input et avec de l'augmentation

   ```sh 
   python3 src/train.py --use_gpu --model resnet --normalize --num_workers 8 --data_augment
   ```

Entrainer un modèle avec normalisation et augmentation des données

   ```sh 
   python3 src/train.py --use_gpu --model resnet --normalize --data_augment--num_workers 8
   ```
   
Generer le csv des resultat a partir d'un modele déjà entraîné (normaliser si le modele était normalisé)

   ```sh  
   python3 src/test.py --paramfile resnet34_2.2/best_model.pt --dir Results --model resnet --normalize --num_workers 8
   ```

# Lien vidéo de la présentation
   ```sh 
   https://youtu.be/TxIvZuXQ3m0
   ```
