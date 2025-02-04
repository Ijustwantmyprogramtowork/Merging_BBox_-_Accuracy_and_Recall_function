# Bounding Box Merging Tool

## Description

Ce script Python est conçu pour manipuler des annotations de bounding boxes stockées dans un fichier JSON. Il permet de fusionner des boîtes englobantes qui se chevauchent ou qui sont proches, selon des seuils prédéfinis, en mettant l'accent sur la détection et la fusion des boîtes correspondant à des objets spécifiques (comme des roues).

---

## Fonctionnalités

### 1. Chargement et sauvegarde des fichiers JSON
- **`open_json_from_file(json_path)`**
  - Charge un fichier JSON depuis le chemin spécifié.
  - Affiche une erreur si le fichier ne peut pas être ouvert.

- **`save_json_to_file(json_data, json_path)`**
  - Sauvegarde un objet JSON dans un fichier au chemin spécifié.
  - Affiche une erreur si l'écriture échoue.

- **`pretty_print(inline_json)`**
  - Affiche un JSON dans un format lisible, avec indentation et tri des clés.

### 2. Extraction des boîtes englobantes
- **`extract_bounding_boxes(bb)`**
  - Extrait les boîtes englobantes d'un fichier JSON contenant des annotations.
  - Renvoie un dictionnaire où chaque clé est le chemin d'une image, et chaque valeur est une liste de boîtes englobantes sous forme de tuples `(x1, y1, x2, y2)`.

### 3. Fusion des boîtes englobantes
- **`merging_axes(bb_all_images, image_path, thresh1=0.01, thresh2=0.015)`**
  - Fusionne les boîtes englobantes qui se chevauchent ou sont proches.
  - Critères de fusion :
    - Les boîtes doivent se chevaucher ou être suffisamment proches (seuil `thresh1`).
    - La nouvelle boîte formée ne doit pas dépasser une certaine aire (seuil `thresh2`).
  - Retourne un dictionnaire des nouvelles boîtes englobantes fusionnées.

- **`verrifying_axes(bb, new_bounding_box)`**
  - Vérifie si des boîtes ont effectivement été fusionnées.
  - Retourne `True` si au moins une fusion a eu lieu, `False` sinon.

### 4. Pipeline principal
- **`main(bb_all_images)`**
  - Traite les boîtes englobantes de toutes les images.
  - Assure que des fusions sont effectuées en itérant jusqu'à 10 fois si nécessaire.
  - Retourne un dictionnaire contenant les annotations mises à jour pour toutes les images.

---

## Utilisation

1. **Chargement des annotations**
   - Le script commence par charger les annotations à partir d'un fichier JSON appelé `annotations.json`.

2. **Fusion des boîtes englobantes**
   - Les boîtes englobantes sont extraites, traitées, et fusionnées si elles respectent les critères définis.

3. **Affichage et sauvegarde**
   - Les annotations finales sont affichées dans un format lisible et sauvegardées dans un nouveau fichier `new_annotations.json`.

---

## Points importants

- **Seuils de fusion** :
  - `thresh1` (distance minimale pour fusion) : par défaut `0.01`.
  - `thresh2` (aire maximale pour fusion) : par défaut `0.015`.

- **Gestion des erreurs** :
  - Le script gère les exceptions liées au chargement et à l'écriture des fichiers JSON.

- **Limitation des itérations** :
  - La boucle de fusion dans la fonction `main` est limitée à 10 itérations pour éviter les boucles infinies.

---

## Exemple d'exécution

1. Assurez-vous que le fichier `annotations.json` contient les annotations au format requis.
2. Exécutez le script :
   ```bash
   python3 bounding_box_merger.py



## Points d'avancement et discussion

Les valeurs des deux seuils (`thresh1` et `thresh2`) ont été déterminées à partir de tests répétitifs effectués sur plusieurs images (voir *Untitled_1.ipynb*). Initialement, une seule valeur pour ces seuils semblait adaptée pour l'image 1, mais les images 2, 3 et 4 nécessitaient des ajustements. L'image 5, quant à elle, a nécessité deux passes dans la boucle pour traiter correctement les trois roues arrière.

Pour résoudre ce problème, une condition a été ajoutée afin de gérer dynamiquement les seuils : tant que de nouvelles boîtes englobantes n'étaient pas générées (c'est-à-dire que le nombre de boîtes restait inchangé par rapport à l'image d'origine), la liste d'annotations était repassée dans la fonction `merge_axes` avec des valeurs de seuil ajustées. Cette méthode s'est avérée efficace.

De plus, une limite de 10 itérations a été intégrée dans la boucle `while` de `merge_axes` pour éviter les boucles infinies, un problème qui se posait notamment avec l'image 4.

