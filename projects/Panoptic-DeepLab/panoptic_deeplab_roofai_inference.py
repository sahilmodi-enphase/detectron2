import time

import cv2
import json
import numpy as np
import os
import re
from tqdm import tqdm
import torch.utils.data as torchdata

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import DefaultTrainer
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer

def create_roofai_panoptic_dataset():
    dataset_folder = "/Users/sahilmodi/Projects/Image_SuperResolution/building-outline-detection/services/roof_line_detection/cache/test_717_baseline"
    image_folder = os.path.join(dataset_folder, "images")
    # pan_seg_gt_folder = os.path.join(dataset_folder, "images")
    # contour_json_file = "/Users/sahilmodi/Projects/Git_Repos/detectron2/datasets/roofai_18k/vgg_till_oct31_outer_contour.json"

    # with open(contour_json_file, "r") as f:
    #     contour_data = json.load(f)

    dataset_list = []

    for filename in os.listdir(image_folder):
        # outer_contour = contour_data[filename]
        dataset_list.append({
            "file_name": os.path.join(image_folder, filename),
            "height": 800,
            "width":  800,
            "image_id": int(re.findall("\d+", filename)[0]),
            # "pan_seg_file_name": os.path.join(pan_seg_gt_folder, filename),
            # "segments_info": [{
            #     "bbox": [
            #         min(outer_contour, key=lambda x: x[0])[0],
            #         min(outer_contour, key=lambda x: x[1])[1],
            #         max(outer_contour, key=lambda x: x[0])[0],
            #         max(outer_contour, key=lambda x: x[1])[1]
            #     ],
            #     "bbox_mode": BoxMode.XYXY_ABS,
            #     "category_id": 0,
            #     "id": 16777215,
            #     "iscrowd": 0,
            # }]
        })

    return dataset_list

def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

if __name__ == "__main__":
    image_dir = "/Users/sahilmodi/Projects/Image_SuperResolution/building-outline-detection/services/roof_line_detection/cache/test_717_baseline/images"
    output_dir = "results/test_717_baseline_" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir)
    config_file = "/Users/sahilmodi/Projects/Git_Repos/detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv_roofai.yaml"
    train_init_checkpoint = "/Users/sahilmodi/Projects/Git_Repos/detectron2/projects/Panoptic-DeepLab/models/model_final.pth"

    cfg = setup(config_file)
    dataset = DatasetFromList(create_roofai_panoptic_dataset(), copy=False)
    mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)
    dataloader = torchdata.DataLoader(
        dataset,
        batch_size=1,
        sampler=InferenceSampler(len(dataset)),
        drop_last=False,
        num_workers=0,
        collate_fn = trivial_batch_collator
    )

    model = DefaultTrainer.build_model(cfg)
    model.to("cpu")
    DetectionCheckpointer(model, save_dir=output_dir).resume_or_load(
        train_init_checkpoint, resume=False
    )
    model.eval()

    for idx, inputs in tqdm(enumerate(dataloader)):

        outputs = model(inputs)
        img = cv2.imread(inputs[0]["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(img, MetadataCatalog.get("cityscapes_fine_panoptic_train"))
        out = visualizer.draw_panoptic_seg(outputs[0]["panoptic_seg"][0], None, alpha=0.2)
        pan_seg_mask = outputs[0]["panoptic_seg"][0].numpy()
        pan_seg_mask_new = np.zeros(pan_seg_mask.shape, dtype=np.uint8)
        pan_seg_mask_new[np.where(pan_seg_mask == 0)] = 255
        cv2.imwrite(os.path.join(output_dir, os.path.basename(inputs[0]["file_name"])), pan_seg_mask_new)
        # x = out.get_image()
        # cv2.imwrite(os.path.join(output_dir, os.path.basename(inputs[0]["file_name"])), cv2.cvtColor(x, cv2.COLOR_BGR2RGB))

