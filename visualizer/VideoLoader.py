from visualizer.visualization import Visualization, NoVisualization
import cv2
import pathlib
import os
import numpy as np
import time

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
        _t0 = time.time()
        ret, frame = self.vcap.read()
        _t1 = time.time() - _t0
        if ret:
            frame_callback(self, frame, self.frame_idx)
            self.viewer.annotate(4, 50, "io_fps {:03.1f}".format(1 / _t1))
            self.frame_idx += 1
            return True
        else:
            return False

class ImageLoader(Visualization):
    def __init__(self, image_dir, update_ms, running_name="", starting_frameid=0, crop_area=None):
        supported_formats = [".png", ".jpg"]

        self.image_filenames = {
            int(idx): os.path.join(image_dir, f)
            for idx, f in enumerate(sorted(os.listdir(image_dir))) if os.path.splitext(f)[-1] in supported_formats}

        if len(self.image_filenames) > 0:
            image = cv2.imread(next(iter(self.image_filenames.values())),
                               cv2.IMREAD_GRAYSCALE)

            self.crop_image = crop_area
            if self.crop_image:
                image_size = (image.shape[0] - self.crop_image[0] - self.crop_image[1], image.shape[1] - self.crop_image[2] - self.crop_image[3])
            else:
                image_size = image.shape
            image_ratio = int(image_size[1]) / int(image_size[0])
            print("IMAGE SIZE: {} RATIO: {}".format(image_size, image_ratio))
        else:
            image_size = None

        if len(self.image_filenames) > 0:
            min_frame_idx = min(self.image_filenames.keys())
            max_frame_idx = max(self.image_filenames.keys())

        else:
            min_frame_idx = 0
            max_frame_idx = 0

        seq_info = {
            "sequence_name": "{} {}".format(os.path.basename(image_dir), running_name) ,
            # "groundtruth": groundtruth,
            "image_size": image_size,
            "min_frame_idx": min_frame_idx,
            "max_frame_idx": max_frame_idx,
            "update_ms": update_ms
        }

        super().__init__(seq_info, update_ms)
        self.frame_idx = starting_frameid

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False
        # _t0 = time.time()
        image = cv2.imread(self.image_filenames[self.frame_idx], cv2.IMREAD_COLOR)
        # _t1 = time.time() - _t0
        if self.crop_image:
            sx = self.crop_image[2]
            sy = self.crop_image[0]
            ex = image.shape[1] - self.crop_image[2] - self.crop_image[3]
            ey = image.shape[0] - self.crop_image[1] - self.crop_image[2]
            frame_callback(self, image[sy:ey, sx:ex], self.frame_idx)
        else:
            frame_callback(self, image, self.frame_idx)
        # self.viewer.annotate(4, 50, "io_fps {:03.1f}".format(1 / _t1))
        self.frame_idx += 1
        return True


class NdImageLoader(NoVisualization):
    def __init__(self, image_dir):
        self.image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        if len(self.image_filenames) > 0:
            image = cv2.imread(next(iter(self.image_filenames.values())),
                               cv2.IMREAD_GRAYSCALE)
            image_size = image.shape
        else:
            image_size = None

        if len(self.image_filenames) > 0:
            min_frame_idx = min(self.image_filenames.keys())
            max_frame_idx = max(self.image_filenames.keys())

        else:
            min_frame_idx = 0
            max_frame_idx = 0

        seq_info = {
            "sequence_name": os.path.basename(image_dir),
            # "groundtruth": groundtruth,
            "image_size": image_size,
            "min_frame_idx": min_frame_idx,
            "max_frame_idx": max_frame_idx,
            "update_ms": 0
        }

        super().__init__(seq_info)

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, None, self.frame_idx)
            self.frame_idx += 1