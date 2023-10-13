import tqdm
import shutil
from pathlib import Path
import numpy as np

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import plotly.io as pio
# # Setup
# Here we define some output paths.

dp_all_images = Path('datasets/sfm_4_run_oct_9_tot_300').resolve()
dp_process_and_outputs = Path('outputs/demo_sfm_4_run_oct_9_tot_300/').resolve()
if dp_process_and_outputs.exists():
    shutil.rmtree(dp_process_and_outputs)
sfm_pairs = dp_process_and_outputs / 'pairs-sfm.txt'
loc_pairs = dp_process_and_outputs / 'pairs-loc.txt'
sfm_dir = dp_process_and_outputs / 'sfm'
features = dp_process_and_outputs / 'features.h5'
matches = dp_process_and_outputs / 'matches.h5'

# Partition all Images into Mapping and Query
dp_mapping = dp_process_and_outputs / 'mapping'
dp_mapping.mkdir(parents=True, exist_ok=True)
dp_query = dp_process_and_outputs / 'query'
dp_query.mkdir(parents=True, exist_ok=True)
fps_all_images = [fp for fp in dp_all_images.glob('*.png') if fp.stem[0] not in ['_', '.']]
num_mapping = 500
num_query = 100
# Random Sample Mapping and Query
fps_mapping = np.random.choice(fps_all_images, num_mapping, replace=False)
fps_query = np.random.choice([fp for fp in fps_all_images if fp not in fps_mapping], num_query, replace=False)
# Copy them into corresponding folder
for fp in tqdm.tqdm(fps_mapping):
    shutil.copy(fp, dp_mapping / fp.name)
for fp in tqdm.tqdm(fps_query):
    shutil.copy(fp, dp_query / fp.name)
# ===============================================


feature_conf = extract_features.confs['disk']
matcher_conf = match_features.confs['disk+lightglue']

# 3D mapping
# First we list the mapping used for mapping. These are all day-time shots of Sacre Coeur.
references = [p.relative_to(dp_mapping).as_posix() for p in dp_mapping.glob('*.png') if p.stem[0] not in ['_', '.']]
print(len(references), "images for mapping")
# plot_images([read_image(images / r) for r in references], dpi=25)

extract_features.main(feature_conf, dp_mapping, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# Then we extract features and match them across image pairs. Since we deal with few mapping, we simply match all pairs
# exhaustively. For larger scenes, we would use image retrieval, as demonstrated in the other notebooks.
model = reconstruction.main(sfm_dir, dp_mapping, sfm_pairs, features, matches, image_list=references)
fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
# fig.show()
pio.write_image(fig, dp_process_and_outputs / "viz_3d_rec.svg")
visualization.visualize_sfm_2d(model, dp_mapping, color_by='visibility', n=2)

#Localization
# Now that we have a 3D map of the scene, we can localize any image. To demonstrate this, we download a night-time image
# from Wikimedia.

url = "https://upload.wikimedia.org/wikipedia/commons/5/53/Paris_-_Basilique_du_Sacr%C3%A9_Coeur%2C_Montmartre_-_panoramio.jpg"
# try other queries by uncommenting their url
# url = "https://upload.wikimedia.org/wikipedia/commons/5/59/Basilique_du_Sacr%C3%A9-C%C5%93ur_%285430392880%29.jpg"
# url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/Sacr%C3%A9_C%C5%93ur_at_night%21_%285865355326%29.jpg"
query = 'query/night.jpg'
# !mkdir -p $mapping/query && wget $url -O $mapping/$query -q

import urllib.request
dp_query = Path(f"{images}/query").resolve()
dp_query.mkdir(exist_ok=True, parents=True)
# Assuming variables url, mapping and query are defined
urllib.request.urlretrieve(url, f"{images}/{query}")

plot_images([read_image(images / query)], dpi=75)

# Again, we extract features for the query and match them exhaustively.
extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)

#We read the EXIF data of the query to infer a rough initial estimate of camera parameters like the focal length.
# Then we estimate the absolute camera pose using PnP+RANSAC and refine the camera parameters.
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

camera = pycolmap.infer_camera_from_image(images / query)
ref_ids = [model.find_image_with_name(r).image_id for r in references]
conf = {
    'estimation': {'ransac': {'max_error': 12}},
    'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
}
localizer = QueryLocalizer(model, conf)
ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
visualization.visualize_loc_from_log(images, query, log, model)

# We visualize the correspondences between the query mapping a few mapping mapping.
# We can also visualize the estimated camera pose in the 3D map.

pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)
# visualize 2D-3D correspodences
inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
# fig.show()
pio.write_image(fig, dp_process_and_outputs / "viz_3d_query.svg")