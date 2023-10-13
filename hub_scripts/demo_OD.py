__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch as pt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np


if __name__ == '__main__':
    # INIT MODEL
    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval().cuda()
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # GET IMAGES
    dp_images = Path("E:\sfm\data\pco_image_front_30_straight\images").resolve()
    fps_img = sorted(dp_images.glob("*.jpg"))
    results = []
    with pt.no_grad():
        for fp in tqdm(fps_img):
            img_ori = Image.open(fp)
            img_4_model = np.array(img_ori)
            # Step 3: Apply inference preprocessing transforms
            batch = [preprocess(img_ori).cuda()]
            prediction = model(batch)[0]
            # OUTPUT
            # Step 4: Use the model and visualize the prediction

            labels = [weights.meta["categories"][i] for i in prediction["labels"]]
            box = draw_bounding_boxes(pt.Tensor(img_4_model).permute(2, 0, 1).byte(), boxes=prediction["boxes"],
                                      labels=labels,
                                      colors="red",
                                      width=4, font_size=30)
            im = to_pil_image(box.detach())
            results.append(im)
    pass
    for im in results:
        im.show()