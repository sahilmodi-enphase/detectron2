import time

import cv2
import os
import re
from tqdm import tqdm

from omegaconf import OmegaConf

from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L, LazyConfig, instantiate
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils.visualizer import Visualizer

def get_roofai_dataset(image_dir):

    image_dict_list = []
    for img_file in os.listdir(image_dir):
        filename = os.path.join(image_dir, img_file)
        image_dict_list.append({
            "file_name": filename,
            "height": 800,
            "width": 800,
            "image_id": re.findall("\d+", filename)[0]
        })

    return image_dict_list

def get_dataloader(image_dir):
    target_image_size = 1024
    testdataloader = L(build_detection_test_loader)(
        dataset=L(get_roofai_dataset)(image_dir=image_dir),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=target_image_size, max_size=target_image_size),
            ],
            image_format="RGB",
        ),
        num_workers=0,
    )

    return testdataloader

if __name__ == "__main__":
    image_dir = "/Users/sahilmodi/Projects/Image_SuperResolution/building-outline-detection/services/roof_line_detection/cache/test_717_baseline/images"
    output_dir = "results/test_717_baseline_" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir)
    config_file = "/Users/sahilmodi/Projects/Git_Repos/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_l_100ep.py"
    train_init_checkpoint = "/Users/sahilmodi/Projects/Git_Repos/detectron2/projects/ViTDet/models/vitdet_l_coco_cascade.pkl"

    cfg = LazyConfig.load(config_file)
    dataloader = instantiate(get_dataloader(image_dir))

    model = instantiate(cfg.model)
    model.to("cpu")
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(train_init_checkpoint)
    model.eval()

    for idx, inputs in tqdm(enumerate(dataloader)):

        outputs = model(inputs)
        img = cv2.imread(inputs[0]["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(img)
        out = visualizer.draw_instance_predictions(outputs[0]["instances"])
        x = out.get_image()
        cv2.imwrite(os.path.join(output_dir, os.path.basename(inputs[0]["file_name"])), cv2.cvtColor(x, cv2.COLOR_BGR2RGB))

