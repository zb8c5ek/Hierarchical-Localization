from database import COLMAPDatabase, blob_to_array
import numpy as np
__author__ = 'Xuan-Li CHEN'
"""
Xuan-Li Chen
Domain: Computer Vision, Machine Learning
Email: chen_alphonse_xuanli(at)outlook.com
"""


def align_all_images_to_a_single_camera(fp_db, cam_id):
    """
    NOT SURE IS WORKING
    Parameters
    ----------
    fp_db
    cam_id

    Returns
    -------

    """
    id_x = cam_id
    db = COLMAPDatabase.connect(fp_db)
    cameras = db.execute("SELECT * FROM cameras;").fetchall()
    camera_params = [blob_to_array(c[4], np.float64) for c in cameras]
    db.execute("UPDATE images SET camera_id = ?;", (id_x,))
    db.commit()
    db.close()


def initialize_db_from_folder_imgs_and_explicit_camera_params(dp_images, fp_db, cam_params):
    pass