# Lancez les commandes sur gpu

1. Entrainer un modèle en normalisant les input et avec de l'augmentation
   ```sh 
    python3 src/train.py --use_gpu --model resnet --normalize --num_workers 8 --data_augment
   ```

3. Réutiliser le meilleur modèle entrainer si le modele ne s'est pas sauvegarder correctement
    ```sh 
    python3 src/best_model.py --use_gpu --model resnet --normalize --log_model ./logs/resnet_13 --num_workers 8

   ```
4. Generer le csv des resultat a partir d'un modele entrainer (normaliser si le modele a etait normalise)
    ```sh  
    python3 src/test.py --paramfile ./logs/resnet_13/best_model.pt --dir result --model resnet --normalize --num_workers 8

   ```
