# #!/usr/bin/env python3
import json
import time
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #


def open_json_from_file(json_path):
    """
    Loads a json from a file path.

    :param json_path: path to the json file
    :return: the loaded json
    """
    try:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
    except:
        print(f"Could not open file {json_path} in json format.")
        raise

    return json_data


def save_json_to_file(json_data, json_path):
    """
    Saves a json to a file.

    :param json_data: the actual json
    :param json_path: path to the json file
    :return:
    """
    try:
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file)
    except:
        print(f"Could not save file {json_path} in json format.")
        raise

    return


def pretty_print(inline_json):
    """
    Prints a json in the command interface in easily-readable format.

    :param inline_json:
    :return:
    """
    print(json.dumps(inline_json, indent=4, sort_keys=True))
    return

def extract_bounding_boxes(bb, score:bool):
    """
    This code returns the bouding boxes of each image as a form of a dict and takes the score as 1 if ground truth

    Parameters
    ----------
    bb : the json file with the bounding boxes 

    Returns
    -------
    dict
        The dict with keys being the path of the image and as value being a list of the bounding boxes in this image
    """
    bb_all_images = {}
    for img in bb['images']:
        image_path = img['location']
        boun_box = []
        for region in img['annotated_regions']:
            region_data = region['region']
            x1, y1, x2, y2 = region_data['xmin'], region_data['ymin'], region_data['xmax'], region_data['ymax']
            if score==True:
                score=region['score']
                boun_box.append((x1, y1, x2, y2, score))
            else:
                boun_box.append((x1,x2,y1,y2, 1))
        bb_all_images[image_path] = boun_box
    return bb_all_images


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a NumPy array of dimensions (n1, 4)
    :param set_2: set 2, a NumPy array of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a NumPy array of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    union = areas_set_1[:, None] + areas_set_2[None, :] - intersection  # (n1, n2)
    return intersection / union  # (n1, n2)

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a NumPy array of dimensions (n1, 4)
    :param set_2: set 2, a NumPy array of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a NumPy array of dimensions (n1, n2)
    """

    # Compute lower and upper bounds
    lower_bounds = np.maximum(set_1[:, None, :2], set_2[None, :, :2])  # (n1, n2, 2)
    upper_bounds = np.minimum(set_1[:, None, 2:], set_2[None, :, 2:])  # (n1, n2, 2)

    # Compute intersection dimensions
    intersection_dims = np.clip(upper_bounds - lower_bounds, a_min=0, a_max=None)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def evaluate_image(bb_gt, bb_pred, threshold, jaccard_overlap=0.5):
    """This function first selects the prediction bouding box with a trust score > threshold
     then  evaluates all the images and makes a matrice of jaccard overlap to select the matching bb
    Then calculates the true positives, flase positives and false negatives.

    Parameters
    ----------
    bb_gt : dict : The dict with the score, and the true bounding boxes
    bb_pred : dict : the dict with the score and the bounding boxes predicted
    threshold : float, evaluates at what score we take into consideration the predicted bounding boxes
    jaccard_overlap : float, The number where we considered two bouding boxes are true positives when intersected
     by default 0.5

    Returns
    -------
    floats, the number of true positives, false positives and false negatives in all the images
    """
    Mat_Fin = []

    for image_path in bb_gt.keys():
        image = Image.open(image_path)
        pred_filtered = [
            (X1, Y1, X2, Y2, score)
            for (X1, Y1, X2, Y2, score) in bb_pred[image_path]
            if score >= threshold 
        ]
        if not pred_filtered:  
            Mat_Fin.append([0, 0, len(bb_gt[image_path])])
            continue
        bb_pred_filtered = np.array([p[:4] for p in pred_filtered])

        matrice_finale = np.zeros((len(bb_gt[image_path]), len(bb_pred_filtered)))

        # Calcul IOU pour toutes les paires
        for i, (x1, y1, x2, y2,score_pred) in enumerate(bb_gt[image_path]):
            for j, (X1, Y1, X2, Y2) in enumerate(bb_pred_filtered):
                score = find_jaccard_overlap(
                    np.array([[x1 * image.size[0], y1 * image.size[1], x2 * image.size[0], y2 * image.size[1]]]),
                    np.array([[X1 * image.size[0], Y1 * image.size[1], X2 * image.size[0], Y2 * image.size[1]]])
                )
                matrice_finale[i, j] = score[0][0]

        # Match pred pour grount truth
        n_gt, n_pred = matrice_finale.shape
        gt_matched = [False] * n_gt
        pred_matched = [False] * n_pred

        for i in range(n_gt):
            max_iou = 0
            match_idx = -1
            for j in range(n_pred):
                if matrice_finale[i, j] > max_iou and matrice_finale[i, j] >= jaccard_overlap:
                    max_iou = matrice_finale[i, j]
                    match_idx = j

            if match_idx != -1:
                gt_matched[i] = True
                pred_matched[match_idx] = True

        true_positives = sum(gt_matched)
        false_positives = sum([not matched for matched in pred_matched])
        false_negatives = sum([not matched for matched in gt_matched])

        Mat_Fin.append([true_positives, false_positives, false_negatives])

    True_Pos = sum([tp[0] for tp in Mat_Fin])
    Fal_Pos = sum([fp[1] for fp in Mat_Fin])
    Fal_Neg = sum([fn[2] for fn in Mat_Fin])

    return True_Pos, Fal_Pos, Fal_Neg



def evaluate_pr_naive(annotations, predictions, N=10, Jaccard_min=0.5):
    """Cette fonction calcule la précision et le recall

    Parameters
    ----------
    annotations : dict: json file des ground truth
    predictions : dict: json file des predictions
    N : int, optional, le nombre d'essais basés sur des thresholds différents by default 10
    Jaccard_min : float, The number where we considered two bouding boxes are true positives when intersected
     by default 0.5

    Returns
    -------
   dict, la precision et le recall regarding the threshold used
    """
    thresholds = np.linspace(0.0, 1.0, N)
    results = []
    gt_boxes=extract_bounding_boxes(annotations, False)
    pred_boxes=extract_bounding_boxes(predictions, True)
    for threshold in thresholds:
        tp, fp, fn = evaluate_image(gt_boxes, pred_boxes, threshold, Jaccard_min)
        precision=(tp/(tp+fp)) if tp+fp>0 else 0
        recall=(tp/(tp+fn)) if tp+fn>0 else 0
        results.append({"precision": precision, "recall": recall, "threshold": float(threshold)})

    return results



def evaluate_pr(annotations, predictions, N=10, Jaccard_min=0.5):
    """ This function tries to combine the evaluate_pr_naive and the evaluate_image for better efficiency

    Parameters
    ----------
    annotations : dict: json file des ground truth
    predictions : dict: json file des predictions
    N : int, optional, le nombre d'essais basés sur des thresholds différents by default 10
    Jaccard_min : float, The number where we considered two bouding boxes are true positives when intersected
     by default 0.5

    Returns
    -------
   dict, la precision et le recall regarding the threshold used
    """
    gt_boxes = extract_bounding_boxes(annotations, False)
    pred_boxes = extract_bounding_boxes(predictions, True)
    thresholds = np.linspace(0.0, 1.0, N)
    
    results = []

    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0  
        for image_path, gt in gt_boxes.items():
            image = Image.open(image_path)
            preds = [(x1, y1, x2, y2) for x1, y1, x2, y2, score in pred_boxes.get(image_path, []) if score >= threshold]

            if not preds:
                fn += len(gt)
                continue
            preds = np.array(preds)
            gt = np.array(gt)[:, :4]  # On ignore le score 
            
            if gt.size == 0:
                fp += len(preds)
                continue
            
            # Calcul de l'IOU entre chaque GT et chaque pred
            iou_matrix = np.zeros((len(gt), len(preds)))
            for i, g in enumerate(gt):
                for j, p in enumerate(preds):
                    iou_matrix[i, j] = find_jaccard_overlap(
                        np.array([[g[0] * image.size[0], g[1] * image.size[1], g[2] * image.size[0], g[3] * image.size[1]]]),
                        np.array([[p[0] * image.size[0], p[1] * image.size[1], p[2] * image.size[0], p[3] * image.size[1]]])
                    )[0, 0]
            # Associer chaque GT au meilleur predicteur
            matched_gt, matched_pred = set(), set()
            for i in range(len(gt)):
                best_match = np.argmax(iou_matrix[i])
                if iou_matrix[i, best_match] >= Jaccard_min:
                    matched_gt.add(i)
                    matched_pred.add(best_match)

            tp += len(matched_gt)
            fp += len(preds) - len(matched_pred)
            fn += len(gt) - len(matched_gt)
        # Calcul de la précision et du rappel
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
        results.append({"precision": precision, "recall": recall, "threshold": float(threshold)})

    return results



if __name__ == '__main__':
    # Load annotations from json file
    groundtruth = open_json_from_file('groundtruth.json')
    predictions = open_json_from_file('predictions.json')

    # Compare evaluate_pr_naive and evaluate_pr
    T0 = time.time()
    evaluate_pr_naive(groundtruth, predictions, N=10, Jaccard_min=0.5)
    T1 = time.time()
    evaluate_pr(groundtruth, predictions, N=10, Jaccard_min=0.5)
    T2 = time.time()
    print(f"Naive version computed in {round(T1-T0, 2)}s")
    print(f"Optimized version computed in {round(T2-T1, 2)}s")


