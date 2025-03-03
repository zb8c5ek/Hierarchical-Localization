from pathlib import Path
from termcolor import colored
from database import COLMAPDatabase, blob_to_array
import pycolmap
import numpy as np
import time
if __name__ == '__main__':
    output_path = Path("outputs").resolve() / "rec_sfm_4_run_oct_9_m0200-q0100_from_300"
    image_dir = Path("datasets/rec_sfm_4_run_oct_9_m0200-q0100_from_300/mapping").resolve()

    output_path.mkdir(exist_ok=True, parents=True)
    database_path = output_path / "database.db"
    # 1. Initialize Camera parameters from Calibration
    # TODO: try initialize the database with explicit camera parameters, to see the effect. both the images and cameras
    #   are set explicitly.
    start_time = time.time()

    # 2. Feature Extraction
    pycolmap.extract_features(database_path, image_dir)
    time_extract_features = time.time() - start_time
    pycolmap.match_exhaustive(database_path)

    # 3. Adding Matches
    time_match_features = time.time() - start_time - time_extract_features
    # Comment:
    #   Here The Data Base Codes Can Be Plugged in So that To Add Matches, hence only the long-term reliable features
    #   are selected. maybe set-up a class, which running in the background, to figure out which object segmentation
    #   are reliable in past experience and rich in the area. only use such features. the estimation shall be good.
    # 2. Add features
    # ASSIGN ALL THE IMAGES TO THE SAME CAMERA
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    time_mapping = time.time() - start_time - time_extract_features - time_match_features
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
    time_output = time.time() - start_time - time_extract_features - time_match_features - time_mapping
    print(colored(f"Time Elapsed: {time.time() - start_time}", 'red'))
    print(colored(f"Time Extract Features: {time_extract_features}", 'green'))
    print(colored(f"Time Match Features: {time_match_features}", 'yellow'))
    print(colored(f"Time Mapping: {time_mapping}", 'blue'))
    print(colored(f"Time Output: {time_output}", 'magenta'))
    # dense reconstruction
    mvs_path = output_path / "mvs"
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    print("Undistorted Images are Saved into: ", mvs_path)
    # pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    # pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
