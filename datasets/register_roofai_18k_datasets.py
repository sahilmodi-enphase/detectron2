import json
import os
import re

from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.structures.boxes import BoxMode

def create_roofai_dataset():
    dataset_folder = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k"
    image_folder = os.path.join(dataset_folder, "images")
    contour_json_file = [os.path.join(dataset_folder, c_file) for c_file in os.listdir(dataset_folder) if "outer_contour" in c_file][0]

    with open(contour_json_file, "r") as f:
        contour_data = json.load(f)

    dataset_list = []

    for filename in os.listdir(image_folder):
        outer_contour = contour_data[filename]
        dataset_list.append({
            "file_name": os.path.join(image_folder, filename),
            "height": 800,
            "width":  800,
            "image_id": int(re.findall("\d+", filename)[0]),
            "annotations": [{
                "bbox": [
                    min(outer_contour, key=lambda x: x[0])[0],
                    min(outer_contour, key=lambda x: x[1])[1],
                    max(outer_contour, key=lambda x: x[0])[0],
                    max(outer_contour, key=lambda x: x[1])[1]
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
                "segmentation": [[x for y in outer_contour for x in y]]
            }]
        })

    return dataset_list

def register_roofai_dataset():

    if "roofai_18k" in DatasetCatalog.list():
        DatasetCatalog.remove("roofai_18k")
    DatasetCatalog.register("roofai_18k", create_roofai_dataset)
    MetadataCatalog.get("roofai_18k").thing_classes = ["roof"]

##############################################################

def create_roofai_train_panoptic_dataset():
    dataset_folder = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k/train"
    image_folder = os.path.join(dataset_folder, "images")
    pan_seg_gt_folder = os.path.join(dataset_folder, "gt_masks")
    contour_json_file = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k/vgg_till_oct31_outer_contour.json"

    with open(contour_json_file, "r") as f:
        contour_data = json.load(f)

    dataset_list = []

    for filename in os.listdir(image_folder):
        outer_contour = contour_data[filename]
        dataset_list.append({
            "file_name": os.path.join(image_folder, filename),
            "height": 800,
            "width":  800,
            "image_id": int(re.findall("\d+", filename)[0]),
            "pan_seg_file_name": os.path.join(pan_seg_gt_folder, "mask_" + filename),
            "segments_info": [{
                "category_id": 0,
                "id": 16777215,
                "iscrowd": 0,
            }, {
                "category_id": 1,
                "id": 0,
                "iscrowd": 0,
            }]
        })

    return dataset_list

def create_roofai_val_panoptic_dataset():
    dataset_folder = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k/val"
    image_folder = os.path.join(dataset_folder, "images")
    pan_seg_gt_folder = os.path.join(dataset_folder, "gt_masks")
    contour_json_file = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k/vgg_till_oct31_outer_contour.json"

    with open(contour_json_file, "r") as f:
        contour_data = json.load(f)

    dataset_list = []

    for filename in os.listdir(image_folder):
        outer_contour = contour_data[filename]
        dataset_list.append({
            "file_name": os.path.join(image_folder, filename),
            "height": 800,
            "width":  800,
            "image_id": int(re.findall("\d+", filename)[0]),
            "pan_seg_file_name": os.path.join(pan_seg_gt_folder, "mask_" + filename),
            "segments_info": [{
                "category_id": 0,
                "id": 16777215,
                "iscrowd": 0,
            }, {
                "category_id": 1,
                "id": 0,
                "iscrowd": 0,
            }]
        })

    return dataset_list

def register_roofai_panoptic_dataset():

    if "roofai_18k_panoptic_train" in DatasetCatalog.list():
        DatasetCatalog.remove("roofai_18k_panoptic_train")
    DatasetCatalog.register("roofai_18k_panoptic_train", create_roofai_train_panoptic_dataset)
    MetadataCatalog.get("roofai_18k_panoptic_train").thing_classes = ["roof"]
    MetadataCatalog.get("roofai_18k_panoptic_train").ignore_label = 255
    MetadataCatalog.get("roofai_18k_panoptic_train").thing_dataset_id_to_contiguous_id = {0: 0}

    if "roofai_18k_panoptic_val" in DatasetCatalog.list():
        DatasetCatalog.remove("roofai_18k_panoptic_val")
    DatasetCatalog.register("roofai_18k_panoptic_val", create_roofai_val_panoptic_dataset)
    MetadataCatalog.get("roofai_18k_panoptic_val").thing_classes = ["roof"]
    MetadataCatalog.get("roofai_18k_panoptic_val").ignore_label = 255
    MetadataCatalog.get("roofai_18k_panoptic_val").thing_dataset_id_to_contiguous_id = {0: 0}


if __name__ == "__main__":
    register_roofai_dataset()
    register_roofai_panoptic_dataset()
