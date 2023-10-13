__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import torch as pt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

import platform
def detect_environment():
    if platform.system() == 'Linux':
        if "microsoft" in platform.uname().release.lower():
            return "WSL"
        else:
            return "Linux"
    elif platform.system() == 'Windows':
        return "Windows"
    else:
        return "Unknown"

# Set Hub Path so that WSL and Windows can share models
import torch
from pathlib import Path

pf = detect_environment()
if pf == "WSL":
    dr_hub = Path("/mnt/d/torch_hub").resolve()
    torch.hub.set_dir(dr_hub.as_posix())
    print("Set hub path to: ", torch.hub.get_dir())
elif pf == "Windows":
    dr_hub = Path("D:/torch_hub").resolve()
    torch.hub.set_dir(dr_hub.as_posix())
    print("Set hub path to: ", torch.hub.get_dir())

# load model
model = pt.hub.load('hustvl/yolop', 'yolop', pretrained=True)

#inference
dp_images = Path("E:\sfm\data\pco_image_front_30_straight\images").resolve()
fps_img = sorted(dp_images.glob("*.jpg"))


with pt.no_grad():
    for fp in tqdm(fps_img):
        img_ori = Image.open(fp)
        img_4_model = np.array(img_ori)
        det_out, da_seg_out, ll_seg_out = model(img_4_model)
        # Step 3: Apply inference preprocessing transforms

pass
