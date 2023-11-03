import time
from pathlib import Path
import multiprocessing
import subprocess
__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""

if __name__ == '__main__':
    # Get the number of cores in the system
    available_cores = multiprocessing.cpu_count() - 2

    # Set paths
    image_dir = Path('/opt/data/GX010243-2').resolve()
    output_path = Path('/opt/data/GX010243-2/colmap_cmd_process').resolve()
    output_path.mkdir(parents=True)
    # Start timer
    start_time = time.time()

    # Set COLMAP command with additional required specs
    # colmap_cmd_help = [
    #     "colmap", "mapper", "--help"]
    # subprocess.run(colmap_cmd_help)
    colmap_cmd = [
        "colmap", "automatic_reconstructor",
        "--workspace_path", str(output_path),
        "--image_path", str(image_dir),
        "--quality", "high",
        "--num_threads", str(available_cores),
        "--single_camera", "1",
        "--use_gpu", "1",
    ]

    # Run COLMAP auto reconstruction
    subprocess.run(colmap_cmd)

    # Print execution time
    print(f"Execution time: {time.time() - start_time} seconds")