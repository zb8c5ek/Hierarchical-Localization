__author__ = 'Xuan-Li CHEN'
"""
Xuan-Li Chen
Domain: Computer Vision, Machine Learning
Email: chen_alphonse_xuanli(at)outlook.com
"""
import tqdm
import shutil
from pathlib import Path
import random

num_mapping = 200
num_query = 100

dp_all_images = Path('datasets/sfm_4_run_oct_9_tot_300/mapping').resolve()
dp_selected_images = Path('/opt/data/rec_sfm_4_run_oct_9_m%04d-q%04d_from_300' % (num_mapping, num_query)).resolve()


# ===============================================

# Partition all Images into Mapping and Query
dp_mapping = dp_selected_images / 'cam0' / 'data'
dp_mapping.mkdir(parents=True, exist_ok=True)
dp_query = dp_selected_images / 'query'
dp_query.mkdir(parents=True, exist_ok=True)
fps_all_images = [fp for fp in dp_all_images.glob('*.png') if fp.stem[0] not in ['_', '.']]

# Random Sample Mapping and Query
fps_mapping = random.sample(fps_all_images, num_mapping)
fps_query = random.sample([fp for fp in fps_all_images if fp not in fps_mapping], num_query)
# Copy them into corresponding folder
for fp in tqdm.tqdm(fps_mapping):
    shutil.copy(fp, dp_mapping / fp.name)
for fp in tqdm.tqdm(fps_query):
    shutil.copy(fp, dp_query / fp.name)

print("Mapping and Query Images are Copied into Corresponding Folders:")
print("Mapping: ", dp_mapping)
print("Query: ", dp_query)
