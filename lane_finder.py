import cv2
import numpy as np
from camera_calibrate import load_camera_calibration

class LaneFinder(object):

    def __init__(self, object_detection_func=lambda image: np.zeros_like(image),
                 camera_calibration_file="./output_images/camera_calibration_pickle.p"):
        camera_matrix, distortion = load_camera_calibration(camera_calibration_file)
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.object_detection_func = object_detection_func

    def process_image(self, image):
        """
        :param image: Image with RGB color channels
        :return: new image with all lane line information
        """

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        undistored_image = cv2.undistort(image_bgr, self.camera_matrix, self.distortion, None, self.camera_matrix)

        object_detect_mask = np.zeros_like(undistored_image)
        object_detect_image = self.object_detection_func(undistored_image)
        object_detect_image_masked = cv2.addWeighted(object_detect_image, 1, object_detect_mask, 0.4, 0)

        return cv2.cvtColor(object_detect_image_masked, cv2.COLOR_BGR2RGB)