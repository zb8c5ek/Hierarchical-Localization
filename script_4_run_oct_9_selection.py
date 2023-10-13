import tqdm
import shutil
from pathlib import Path
import numpy as np
import random
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import plotly.io as pio
# # Setup
# Here we define some output paths.

dp_all_images = Path('/opt/datasets/sfm_4_run_oct_9').resolve()
dp_selected_images = Path('datasets/rec_sfm_4_run_oct_9_mapping_1000').resolve()
dp_process_and_outputs = Path('outputs/demo_sfm_4_run_oct_9_tot_300/').resolve()
if dp_process_and_outputs.exists():
    shutil.rmtree(dp_process_and_outputs)
# ===============================================

# Partition all Images into Mapping and Query
dp_mapping = dp_selected_images / 'mapping'
dp_mapping.mkdir(parents=True, exist_ok=True)
dp_query = dp_selected_images / 'query'
dp_query.mkdir(parents=True, exist_ok=True)
fps_all_images = [fp for fp in dp_all_images.glob('*.png') if fp.stem[0] not in ['_', '.']]
num_mapping = 1000
num_query = 100
# Random Sample Mapping and Query
fps_mapping = random.sample(fps_all_images, num_mapping)
fps_query = random.sample([fp for fp in fps_all_images if fp not in fps_mapping], num_query)
# Copy them into corresponding folder
for fp in tqdm.tqdm(fps_mapping):
    shutil.copy(fp, dp_mapping / fp.name)
for fp in tqdm.tqdm(fps_query):
    shutil.copy(fp, dp_query / fp.name)
# ===============================================