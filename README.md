# Lancez les commandes sur gpu

1. Entrainer un modèle en normalisant les input et avec de l'augmentation
   ```sh 
    python3 models/train.py --use_gpu --model FancyCNN --normalize --num_workers 4 --data_augment
   ```

3. Réutiliser le meilleur modèle entrainer pour une architecture et des paramètres spécifiques, et le sauvegarder (a faire avant le test)
    ```sh 
    python3 models/best_model.py --use_gpu --model fancyCNN --normalize

     

   ```
4. Generer le csv des resultat a partir d'un modele entrainer (normaliser si le modele a etait normalise)
    ```sh  
    python3 models/test.py --paramfile ./logs/fancyCNN_9/best_model.pt --dir result --model fancyCNN --normalize

   ```

