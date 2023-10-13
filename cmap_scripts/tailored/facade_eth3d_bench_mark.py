from pathlib import Path
import subprocess
import multiprocessing
import time
__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""


def process_dataset(dp_dataset, fp_colmap_exe, dp_workspace, use_gpu=True):
    print("Processing dataset:", dp_dataset.stem)
    num_threads = multiprocessing.cpu_count() - 2
    # Find undistorted parameters of first camera and initialize all images with it.
    fp_cameras = dp_dataset / 'dslr_calibration_undistorted' / 'cameras.txt'
    with open(fp_cameras, "r") as fid:
        for line in fid:
            if not line.startswith("#"):
                first_camera_data = line.split()
                camera_model = first_camera_data[1]
                assert camera_model == "PINHOLE"
                camera_params = first_camera_data[4:]
                assert len(camera_params) == 4
                break

    # Count the number of expected images in the GT.
    expected_num_images = 0
    fp_images = dp_dataset / 'dslr_calibration_undistorted' / 'images.txt'
    with open(fp_images, "r") as fid:
        for line in fid:
            if not line.startswith("#") and line.strip():
                expected_num_images += 1
    # Each image uses two consecutive lines.
    assert expected_num_images % 2 == 0
    expected_num_images /= 2

    # Run automatic reconstruction pipeline.
    start_recon_time = time.time()
    subprocess.check_call(
        [
            fp_colmap_exe.as_posix(),  # colmap path is just the colmap.BAT path
            "automatic_reconstructor",
            "--image_path",
            (dp_dataset / 'images').as_posix(),
            "--workspace_path",
            dp_workspace.as_posix(),
            "--use_gpu",
            "1" if use_gpu else "0",
            "--num_threads",
            str(num_threads),
            "--quality",
            "high",
            "--camera_model",
            "PINHOLE",
            "--single_camera",
            str(1),     # Whether images are captured by single camera or not
            # "--camera_params",
            # ",".join(camera_params),
        ],
        cwd=dp_workspace.as_posix(),
    )
    print("Reconstruction time:", time.time() - start_recon_time)
    # Compare reconstructed model to GT model.
    subprocess.check_call(
        [
            fp_colmap_exe.as_posix(),
            "model_comparer",
            "--input_path1",
            "sparse/0",
            "--input_path2",
            (dp_dataset / "dslr_calibration_undistorted/").as_posix(),
            "--output_path",
            ".",
            "--alignment_error",
            "proj_center",
            "--max_proj_center_error",
            str(0.1),
        ],
        cwd=dp_workspace.as_posix(),
    )

    # Ensure discrepancy between reconstructed model and GT is small.
    # check_small_errors_or_exit(
    #     dataset_name,
    #     args.max_rotation_error,
    #     args.max_proj_center_error,
    #     expected_num_images,
    #     os.path.join(workspace_path, "errors.csv"),
    # )


def check_small_errors_or_exit(
    dataset_name,
    max_rotation_error,
    max_proj_center_error,
    expected_num_images,
    errors_csv_path,
):
    print(f"Evaluating errors for {dataset_name}")

    error = False
    with open(errors_csv_path, "r") as fid:
        num_images = 0
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            rotation_error, proj_center_error = map(float, line.split(","))
            num_images += 1
            if rotation_error > max_rotation_error:
                print("Exceeded rotation error threshold:", rotation_error)
                error = True
            if proj_center_error > max_proj_center_error:
                print("Exceeded projection center error threshold:", proj_center_error)
                error = True

    if num_images != expected_num_images:
        print("Unexpected number of images:", num_images)
        error = True


def main():
    dp_facade_imgs = Path("E:\datasets\eth3d/facade/facade_undistorted").resolve()
    fp_colmap_exe = Path("E:/nerf_and_colmap\COLMAP-3.8-windows-cuda/COLMAP.bat").resolve()
    dp_work_space = Path("../../colmap_work_space/facade").resolve()
    dp_work_space.mkdir(exist_ok=True)
    process_dataset(
        dp_dataset=dp_facade_imgs, fp_colmap_exe=fp_colmap_exe, dp_workspace=dp_work_space
    )


if __name__ == "__main__":
    main()
