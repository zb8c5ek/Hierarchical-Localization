from pathlib import Path
import pycolmap

if __name__ == '__main__':
    output_path = Path("../outputs").resolve() / "colmap_sfm_4_run_oct_9_50_from_300_single_db_test"
    image_dir = Path("../datasets/rec_sfm_4_run_oct_9_50_from_300/query").resolve()

    output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)

    # Comment:
    #   Here The Data Base Codes Can Be Plugged in So that To Add Matches, hence only the long-term reliable features
    #   are selected. maybe set-up a class, which running in the background, to figure out which object segmentation
    #   are reliable in past experience and rich in the area. only use such features. the estimation shall be good.

    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
