from pathlib import Path
from database import COLMAPDatabase, blob_to_array
import pycolmap
import numpy as np

if __name__ == '__main__':
    output_path = Path("outputs").resolve() / "colmap_sfm_4_run_oct_9_80_from_300"
    image_dir = Path("datasets/rec_sfm_4_run_oct_9_m80-q30_from_300/mapping").resolve()

    output_path.mkdir(exist_ok=True, parents=True)
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    # TODO: try initialize the database with explicit camera parameters, to see the effect. both the images and cameras
    #   are set explicitly.
    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)

    # Comment:
    #   Here The Data Base Codes Can Be Plugged in So that To Add Matches, hence only the long-term reliable features
    #   are selected. maybe set-up a class, which running in the background, to figure out which object segmentation
    #   are reliable in past experience and rich in the area. only use such features. the estimation shall be good.
    # EXTRACT ALL CAMERAS
    # id_x = 1
    # db = COLMAPDatabase.connect(database_path)
    # cameras = db.execute("SELECT * FROM cameras;").fetchall()
    # camera_params = [blob_to_array(c[4], np.float64) for c in cameras]
    # db.execute("UPDATE images SET camera_id = ?;", (id_x,))
    # db.commit()
    # db.close()
    # ASSIGN ALL THE IMAGES TO THE SAME CAMERA
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    map_keys = maps.keys()
    if len(map_keys) == 0:
        raise Exception("No Map is Reconstructed")
    else:
        print(f"{len(map_keys)} Maps are Reconstructed")
    # for key in map_keys:
        for key in map_keys:
            maps[key].write(output_path)
            maps[key].export_PLY(output_path / ("rec_map_%i.ply" % key))
            print(f"Map {key} is written to {output_path}")

    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
