from moviepy.editor import VideoFileClip
from object_detect_yolo import YoloDetector
from time import time
import cv2
import pickle
import numpy as np

class LaneFinder(object):

    def __init__(self, object_detection_func=lambda image: np.zeros_like(image),
                 camera_calibration_file="camera_calibration_pickle.p"):
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

if __name__ == "__main__":
    def remove_mp4_extension(file_name):
        return file_name.replace(".mp4", "")
    def load_camera_calibration(file_name):
        data = pickle.load(open(file_name, "rb"))
        return data["mtx"], data["dist"]

    start = time()
    yolo = YoloDetector()
    lane_finder = LaneFinder(object_detection_func=yolo.process_image_array)
    video_file = 'FHD0121.mp4'#specific your mp4 video file here
    clip = VideoFileClip(video_file, audio=False)
    t_start = 0
    t_end = 0
    if t_end > 0.0:
        clip = clip.subclip(t_start=t_start, t_end=t_end)
    else:
        clip = clip.subclip(t_start=t_start)

    clip = clip.fl_image(lane_finder.process_image)
    clip.write_videofile("{}_output.mp4".format(remove_mp4_extension(video_file)), audio=False)
    yolo.shutdown()#comment out this line if you want to run this program multiple times
    
    end = time()
    print("time taken to analyse this video is: ", end-start, "s")