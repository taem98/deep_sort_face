from visualizer.visualization import Visualization
import cv2
import pathlib
import os
import numpy as np

class VideoLoader(Visualization):
    def __init__(self, videopath, update_ms):
        if not os.path.isfile(videopath):
            raise FileNotFoundError("Video file not found")
        # video_name = pathlib.PurePath(videopath)
        self.vcap = cv2.VideoCapture(videopath)
        if self.vcap.isOpened() == True:
            width = int(self.vcap.get(3))  # float
            height = int(self.vcap.get(4))  # float
        else:
            raise Exception("Could not open video")
        seq_info = {
            "sequence_name": os.path.basename(videopath),
            # "groundtruth": groundtruth,
            "image_size": (height, width),
            "min_frame_idx": 0,
            "max_frame_idx": 0,
            "update_ms": update_ms
        }
        # self.frame_idx
        super().__init__(seq_info, update_ms)

    # def run(self, frame_callback):
    #     self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        # while True:
        ret, frame = self.vcap.read()
        if ret:
            frame_callback(self, frame, self.frame_idx)
            self.frame_idx += 1
            return True
        else:
            return False
