import json
import os
import re

from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts, MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures.boxes import BoxMode

from .COCO.cascade_mask_rcnn_vitdet_l_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

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

    if "roofai_18k" not in DatasetCatalog.list():
        DatasetCatalog.register("roofai_18k", create_roofai_dataset)
        MetadataCatalog.get("roofai_18k").thing_classes = ["roof"]

##############################################################

train.init_checkpoint = (
    "/Users/sahilmodi/Projects/Git_Repos/detectron2/projects/ViTDet/models/vitdet_l_coco_cascade.pkl"
)
train.device = "cpu"
train.amp.enabled = False

model.roi_heads.num_classes = 1

register_roofai_dataset()
dataloader.train.dataset = L(get_detection_dataset_dicts)(names="roofai_18k")
dataloader.test.dataset = L(get_detection_dataset_dicts)(names="roofai_18k", filter_empty=False)
dataloader.train.num_workers = 0
dataloader.test.num_workers = 0
dataloader.train.total_batch_size = 2
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir=train["output_dir"]
)