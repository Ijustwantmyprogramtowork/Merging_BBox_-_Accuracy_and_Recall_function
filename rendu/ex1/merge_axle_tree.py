import json
import numpy as np
import pprint

def open_json_from_file(json_path):
    """Loads a JSON from a file path."""
    try:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
    except:
        print(f"Could not open file {json_path} in JSON format.")
        raise
    return json_data


def save_json_to_file(json_data, json_path):
    """Saves a JSON to a file."""
    try:
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
    except:
        print(f"Could not save file {json_path} in JSON format.")
        raise


def pretty_print(inline_json):
    """Prints a JSON in an easily-readable format."""
    print(json.dumps(inline_json, indent=4, sort_keys=True))


def extract_bounding_boxes(bb):
    """This code returns the bouding boxes of each image as a form of a dict

    Parameters
    ----------
    bb : the json file with the bounding boxes 

    Returns
    -------
    dict
        The dict with keys being the path of the image and as value being a list of the bounding boxes in this image"""
    bb_all_images = {}
    valid_classes = {"car", "other", "single_axle", "grouped_axles"}  # Only keep these classes

    for img in bb['images']:
        image_path = img['location']
        bboxes = []
        
        for region in img['annotated_regions']:
            region_data = region['region']
            label = region['tags'][0]

            if label not in valid_classes:
                continue  # Ignore irrelevant categories

            x1, y1, x2, y2 = region_data['xmin'], region_data['ymin'], region_data['xmax'], region_data['ymax']
            bboxes.append({"label": label, "bbox": [x1, y1, x2, y2]})

        bb_all_images[image_path] = bboxes
    return bb_all_images


def merging_axes(bb_all_images, image_path, thresh1=0.01, thresh2=0.015):
    """This function fist calculates if two bouding boxes are overlapping and if they are close, then is mergint them together.

    Parameters
    ----------
    bb : List : The list of the bouding boxes related to this image
    image_path: str: The image we want to analyse
    thresh1 : float, 
        Is how close the boxes should be to be merged, by default 0.01
    thresh2 : float, optional
        the area of the bouding box to not merge together two big bounding boxes ( we only want the wheels), by default 0.015

    Returns
    -------
    Dict: of the new bouding boxes merged if they are"""
    bb = bb_all_images
    new_bounding_box = {}
    merged_indices = set()

    for i in range(len(bb)):
        bbox_i = bb[i]
        (X1, Y1, X2, Y2) = bbox_i["bbox"]
        label_i = bbox_i["label"]

        for j in range(len(bb)):
            if i == j:
                continue

            bbox_j = bb[j]
            (x1, y1, x2, y2) = bbox_j["bbox"]
            label_j = bbox_j["label"]
            if label_i not in {"single_axle", "grouped_axles"} or label_j not in {"single_axle", "grouped_axles"}:
                continue
            if ((X1 - x1 < 0) or (X2 - x2 < 0) or (Y1 - y1 < 0) or (Y2 - y2 < 0)) and (
                (abs(X1 - x2) < thresh1) or (abs(X2 - x1) < thresh1) or 
                (abs(Y1 - y2) < thresh1) or (abs(Y2 - y1) < thresh1)):

                # Merge boxes
                new_x1 = min(X1, x1)
                new_x2 = max(X2, x2)
                new_y1 = min(Y1, y1)
                new_y2 = max(Y2, y2)

                # Ensure the merged box is not too large
                if (new_x1 - new_x2) * (new_y1 - new_y2) < thresh2:
                    key = (new_x1, new_y1, new_x2, new_y2)
                    new_bounding_box[key] = {"label": "grouped_axles", "bbox": [new_x1, new_y1, new_x2, new_y2]}
                    merged_indices.add(i)
                    merged_indices.add(j)

    # Keeping normal unmerged boxes
    for k in range(len(bb)):
        if k not in merged_indices:
            bbox = bb[k]
            key = tuple(bbox["bbox"])
            new_bounding_box[key] = {"label": bbox["label"], "bbox": list(bbox["bbox"])}

    return new_bounding_box


def verifying_axes(bb, new_bounding_box):
    """This function is to verrify that some of the bouding boxes were merged together ( or else the previous code didn't have any effect )

    Parameters
    ----------
    bb : List: The list of the original bounding boxes
    new_bounding_box : dict :The dict of the new bounding boxes

    Returns
    -------
    bool"""
    return len(new_bounding_box) < (len(bb) - 1)


def main(bb_all_images):
    """This function returns the new annotation dict and makes sure that at least two bb were merged for final output

    Parameters
    ----------
    bb_all_images : dict
        The dict of all bounding boxes related to the image

    Returns
    -------
    dict : the final annotation dict for all images"""
    final_annotations = {}

    for image_path, bb in bb_all_images.items():
        print(f"Processing {image_path} . . .")
        new_bounding_box = merging_axes(bb, image_path)

        i = 0
        while not verifying_axes(bb, new_bounding_box) and i < 10:
            new_bounding_box = merging_axes(list(new_bounding_box.values()), image_path, 0.05, 0.025)
            i += 1

        # Convertir en format Json
        final_annotations[image_path] = [
            {"category": bbox["label"], "region": {"xmin": bbox["bbox"][0], "ymin": bbox["bbox"][1], 
                                                   "xmax": bbox["bbox"][2], "ymax": bbox["bbox"][3]}}
            for bbox in new_bounding_box.values()
        ]

    return final_annotations


if __name__ == '__main__':
    bb_json = open_json_from_file('annotations.json')
    bb = extract_bounding_boxes(bb_json)
    json_data = main(bb)
    pretty_print(json_data)
    save_json_to_file(json_data, 'new_annotations.json')
