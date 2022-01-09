# Lancez les commandes sur gpu

1. Prétraiter les données
    ```sh 
    python3 models/data_pre/do.py
    ```
2. Entrainer un modèle
   ```sh 
    python3 models/train.py --use_gpu --model fancyCNN
   ```

3. Réutiliser le meilleur modèle entrainer pour une architecture et des paramètres spécifiques
    ```sh 
    python3 models/best_model.py --use_gpu --model fancyCNN

    python3 models/test.py --paramfile ./logs/fancyCNN_9/best_model.pt --dir result --model fancyCNN

   ```

Pour ajouter un modèle le mettre dans ann puis l'ajouter a l'argpaser de train.py.

J'ai pas encore fais le test.py mais c'est rapide