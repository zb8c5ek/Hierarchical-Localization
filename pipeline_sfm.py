__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
from pathlib import Path
from matplotlib import pyplot as plt
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval

data_stem = "sfm_4_run_oct_9_tot_300"

images = Path('datasets/%s/images/' % data_stem).resolve()

outputs = Path('outputs/%s/' % data_stem).resolve()
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'

retrieval_conf = extract_features.confs['netvlad']
# feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superglue']
feature_conf = extract_features.confs['disk']
matcher_conf = match_features.confs['disk+lightglue']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
visualization.visualize_sfm_2d(model, images, color_by='track_length', n=5)
visualization.visualize_sfm_2d(model, images, color_by='depth', n=5)
plt.savefig(outputs / "rec_2d.png")
