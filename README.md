Clonage du repo git

   ```sh 
    git clone https://gitlab-student.centralesupelec.fr/hamza.benslimane/3md4040-challenge.git
    cd 3md4040-challenge
   ```
Lancer le train avec main.py

   ```sh 
    python3 src/main.py train PATH_TO_TRAINING_SET
   ```
Lancer le test avec main.py

   ```sh 
    python3 src/main.py test PATH_TO_CHECKPOINT PATH_TO_TEST_SET
   ```

Vous trouverez le fichier csv (résultat du test) dans le dossier Results

# Lancez les commandes sur gpu (avec paramétrage)

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
    python3 src/test.py --paramfile ./logs/resnet34_2.2/best_model.pt --dir Results --model resnet --normalize --num_workers 8

   ```
