# Évaluation de la Précision et du Rappel pour la Détection d'Objets  

##  Introduction  
Ce projet permet d'évaluer la **précision** et le **rappel** d'un modèle de détection d'objets en comparant des **annotations réelles** (ground truth) à des **prédictions** générées par un modèle. Deux implémentations sont disponibles :  
- **Version naïve** (`evaluate_pr_naive`) : méthode simple mais lente.  
- **Version optimisée** (`evaluate_pr`) : améliore les performances en réduisant le nombre de calculs inutiles.  

##  Fonctionnalités  
 - Chargement et sauvegarde des fichiers JSON  
 - Extraction des **bounding boxes** des annotations et prédictions  
 - Calcul de **l'Intersection over Union (IoU)** entre boîtes de détection  
 - Évaluation de la **précision** et du **rappel** sur plusieurs seuils de confiance  
 - Comparaison entre les versions **naïve** et **optimisée**  

## Structure du Code  
- `open_json_from_file(json_path)`: Charge un fichier JSON  
- `save_json_to_file(json_data, json_path)`: Sauvegarde un JSON  
- `extract_bounding_boxes(bb, score)`: Extrait les bounding boxes des fichiers JSON  
- `find_jaccard_overlap(set_1, set_2)`: Calcule l'IOU entre deux ensembles de boîtes  
- `evaluate_image(bb_gt, bb_pred, threshold, jaccard_overlap)`: Évalue une image spécifique  
- `evaluate_pr_naive(annotations, predictions, N, Jaccard_min)`: Version lente  
- `evaluate_pr(annotations, predictions, N, Jaccard_min)`: Version optimisée  

## Optimisation  
La **version optimisée** améliore les performances en :  
**Évitant des boucles inutiles** en vectorisant certaines opérations  
**Réduisant la taille des matrices** pour économiser de la mémoire  
**Filtrant les prédictions** avant d'effectuer les comparaisons 

Heureusement, on voit sur les résultats ( Untitled-1.ipynb ) que la version non optimisée est plus lente avec 0.43s de compute time comparé à 0.32s pour la version optimisée.

