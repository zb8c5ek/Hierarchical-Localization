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

images = Path('datasets/sacre_coeur').resolve()
outputs = Path('outputs/demo/').resolve()
if outputs.exists():
    shutil.rmtree(outputs)
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'
# sfm_dir.mkdir(parents=True, exist_ok=True)
# features.mkdir(parents=True, exist_ok=True)
# matches.mkdir(parents=True, exist_ok=True)

feature_conf = extract_features.confs['disk']
matcher_conf = match_features.confs['disk+lightglue']

# 3D mapping
# First we list the images used for mapping. These are all day-time shots of Sacre Coeur.
references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
print(len(references), "mapping images")
plot_images([read_image(images / r) for r in references], dpi=25)

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# Then we extract features and match them across image pairs. Since we deal with few images, we simply match all pairs
# exhaustively. For larger scenes, we would use image retrieval, as demonstrated in the other notebooks.
model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
# fig.show()
pio.write_image(fig, outputs / "viz_3d_rec.svg")
visualization.visualize_sfm_2d(model, images, color_by='visibility', n=2)

#Localization
# Now that we have a 3D map of the scene, we can localize any image. To demonstrate this, we download a night-time image
# from Wikimedia.

url = "https://upload.wikimedia.org/wikipedia/commons/5/53/Paris_-_Basilique_du_Sacr%C3%A9_Coeur%2C_Montmartre_-_panoramio.jpg"
# try other queries by uncommenting their url
# url = "https://upload.wikimedia.org/wikipedia/commons/5/59/Basilique_du_Sacr%C3%A9-C%C5%93ur_%285430392880%29.jpg"
# url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/Sacr%C3%A9_C%C5%93ur_at_night%21_%285865355326%29.jpg"
query = 'query/night.jpg'
# !mkdir -p $images/query && wget $url -O $images/$query -q

import urllib.request
dp_query = Path(f"{images}/query").resolve()
dp_query.mkdir(exist_ok=True, parents=True)
# Assuming variables url, images and query are defined
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

# We visualize the correspondences between the query images a few mapping images.
# We can also visualize the estimated camera pose in the 3D map.

pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)
# visualize 2D-3D correspodences
inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
# fig.show()
pio.write_image(fig, outputs / "viz_3d_query.svg")