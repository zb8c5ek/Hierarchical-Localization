from pathlib import Path
import pycolmap

if __name__ == '__main__':
    output_path = Path("../outputs").resolve() / "colmap_sfm_4_run_oct_9_tot_300"
    image_dir = Path("../datasets/sfm_4_run_oct_9_tot_300/mapping").resolve()

    output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)

    # Here The Data Base Codes Can Be Plugged in So that To Add Matches

    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
