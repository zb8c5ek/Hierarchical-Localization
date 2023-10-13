__author__ = 'Xuanli CHEN'

import torch
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path

sam_checkpoint = Path("E:\SAM\models\sam_vit_h_4b8939.pth").resolve()
model_type = "vit_h"
device = "cuda"


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


if __name__ == '__main__':
    # INIT MODEL
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # GET IMAGES
    dp_images = Path("E:\sfm\data\pco_image_front_30_straight\images").resolve()
    fps_img = sorted(dp_images.glob("*.jpg"))
    with torch.no_grad():
        for fp in tqdm(fps_img):
            img_ori = Image.open(fp)
            img_4_model = np.array(img_ori)

            masks = mask_generator.generate(img_4_model)

            # OUTPUT
            print(len(masks))
            print(masks[0].keys())
            plt.clf()
            plt.figure(figsize=(20, 12))
            plt.imshow(img_4_model)
            show_anns(masks)
            plt.axis('off')
            plt.show()
            plt.close()
