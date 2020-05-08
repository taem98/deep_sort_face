'''
apply multiprocess loading to speed up loading
just pseudo improvement on
'''

from queue import Queue, Full, Empty
import numpy as np
import time
import cv2
import os, colorsys
from threading import Thread

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class ThreadState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Init = 1
    Running = 2
    Finished = 3

class PoolLoader(object):
    def __init__(self, poolsize = 50, crop_area=None):

        import socket
        self.hostname = socket.gethostname()

        self.image = None

        self._color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.thickness = 1.5
        self.crop_image = crop_area


        self._reading_queue = Queue(poolsize)
        self._display_queue = Queue(poolsize)
        self._detection_queue = Queue(poolsize)
        self._tracker_queue = Queue(poolsize)
        self._terminate = False
        self._loading_index = 0

        self.frame_index = 0

        self.min_frame_idx = 0
        self.max_frame_idx = 0

        self._loadingState = ThreadState.Init
        self._displayState = ThreadState.Init

        self.last_proctime = 1

        self._textsize = 1

        self.vcap = None
        self.image_filenames = None

        #   init the thread
        self._loading_thread = Thread(target=self._load_single_image, args=())
        self._loading_thread.daemon = True
        self._loading_thread.start()
        # this is not thread safe so we only call it onetime
        self._display_thread = Thread(target=self._display_image, args=())
        self._display_thread.daemon = True
        self._display_thread.start()


    def load(self, seq_info):
        # load list of image
        while self._displayState == ThreadState.Running or self._loadingState == ThreadState.Running:
            time.sleep(0.01)
        self.vcap = None
        self.image_filenames = None
        self.seq_info = seq_info

        if self.seq_info["input_type"] == "video":
            # if not os.path.isfile(self.seq_info["video_path"]):
            #     pass
                # raise FileNotFoundError("Video file not found")
            # video_name = pathlib.PurePath(videopath)
            self.vcap = cv2.VideoCapture(self.seq_info["video_path"])
            if self.vcap.isOpened():
                width = int(self.vcap.get(3))  # float
                height = int(self.vcap.get(4))  # float
                aspect_ratio = height / width
                image_size = (width, height)
            else:
                raise Exception("Could not open video {}".format(self.seq_info["video_path"]))

            self.window_name = "%s Video %s" % (self.hostname, self.seq_info["video_path"])
        else:
            supported_formats = [".png", ".jpg"]
            self.image_filenames = {
                int(idx): os.path.join(seq_info['imgdir'], f)
                for idx, f in enumerate(sorted(os.listdir(seq_info['imgdir']))) if
                os.path.splitext(f)[-1] in supported_formats}

            if len(self.image_filenames) > 0:
                image = cv2.imread(next(iter(self.image_filenames.values())),
                                   cv2.IMREAD_GRAYSCALE)

                if self.crop_image:
                    image_size = (image.shape[0] - self.crop_image[0] - self.crop_image[1],
                                  image.shape[1] - self.crop_image[2] - self.crop_image[3])
                else:
                    image_size = image.shape
                aspect_ratio = int(image_size[0]) / int(image_size[1])

            else:
                image_size = None

            if len(self.image_filenames) > 0:
                self.min_frame_idx = min(self.image_filenames.keys())
                self.max_frame_idx = max(self.image_filenames.keys())

            self.window_name = "%s Figure %s" % (self.hostname, self.seq_info["sequence_name"])

        print("IMAGE SIZE: {} RATIO: {}".format(image_size, aspect_ratio))
        #     to display
        try:
            self._image_shape = seq_info['image_shape'], int(seq_info['image_shape'] * aspect_ratio)
        except KeyError:
            self._image_shape = 1024, int(aspect_ratio * 1024)
        # self._image_shape = 1024, int(aspect_ratio * 1024)

        # set the flag to load and display image
        self._loading_index = 0
        self.frame_index = 0
        self._displayState = ThreadState.Running
        self._loadingState = ThreadState.Running


    def _load_single_image(self):
        while True:
            if self._terminate:
            #     this mean we should stop the current thread now
                self._loadingState = ThreadState.Finished
                return

            if self._reading_queue.full() or self._loadingState == ThreadState.Init \
                    or self._loadingState == ThreadState.Finished:
                time.sleep(0.01)
                continue


            #  we will load and do any preprocessing here
            if self.vcap:
                ret, image = self.vcap.read()
                if not ret:
                    if self._loadingState == ThreadState.Running:
                        self._loadingState = ThreadState.Finished
                        self.vcap.release()
                        self.vcap = None
                        print('\n Finish load {} from {}'.format(self.frame_index, self.seq_info["video_path"]))
                else:
                    if self.crop_image:
                        sx = self.crop_image[2]
                        sy = self.crop_image[0]
                        ex = image.shape[1] - self.crop_image[3]
                        ey = image.shape[0] - self.crop_image[2]

                        self._reading_queue.put_nowait(image[sy:ey, sx:ex])
                    else:
                        self._reading_queue.put_nowait(image)
            else:
                if self._loading_index > self.max_frame_idx:
                    if self._loadingState == ThreadState.Running:
                        self._loadingState = ThreadState.Finished
                else:
                    image = cv2.imread(self.image_filenames[self._loading_index], cv2.IMREAD_COLOR)
                    self._loading_index += 1

                    if self.crop_image:
                        sx = self.crop_image[2]
                        sy = self.crop_image[0]
                        ex = image.shape[1] - self.crop_image[3]
                        ey = image.shape[0] - self.crop_image[2]

                        self._reading_queue.put_nowait(image[sy:ey, sx:ex])
                    else:
                        self._reading_queue.put_nowait(image)


    def read(self):
        '''
        Read the image in queue and also put this image to display queue
        :return:
        '''

        while True:
            try:
                img = self._reading_queue.get_nowait()
                break
            except Empty:
                if self._loadingState == ThreadState.Finished: # this mean loading thread has already finish loading the whole sequence
                    raise Exception("Finished")
                else:
                    time.sleep(0.01)
        while True:
            try:
                self._display_queue.put_nowait(img)
                break
            except Full:
                time.sleep(0.01)

        # if self._is_not_loading:
        self._reading_queue.task_done()
        self.frame_index += 1
        print("\r {}/{} Process FPS {} Load {} Playback {} \t\t".format(self.frame_index - 1, self.max_frame_idx,
                                                                1 / self.last_proctime,
                                                                self._reading_queue.qsize(),
                                                                 self._display_queue.qsize(),
                                                                 ), end='')
        return self.frame_index - 1, img

    def rectangle(self, x, y, w, h, label=None, pos=0):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.
        pos : int
            relative position in rectangle 0: top-left, 1: top-right, 2: bot-left, 3: bot-right
        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        if label is not None:
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, self._textsize, self.thickness)
            if pos == 0:
                center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
                label_pt1 = pt1
                label_pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                            text_size[0][1]
            elif pos == 1:
                center = pt2[0] - text_size[0][0], pt1[1] + 5 + text_size[0][1]
                label_pt1 = pt2[0], pt1[1]
                label_pt2 = pt2[0] - 10 - text_size[0][0], pt1[1] + 10 + \
                            text_size[0][1]

            cv2.rectangle(self.image, label_pt1, label_pt2, self._color, -1)
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        self._textsize, (255, 255, 255), self.thickness)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("color must be tuple of 3")
        self._color = tuple(int(c) for c in value)

    def draw_detections(self, detections):
        self.thickness = 2
        self.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.rectangle(*detection.tlwh, label=str(int(detection.label)), pos=1)

    def queue_detection(self, detections):
        while self.seq_info["show_detections"]:
            try:
                self._detection_queue.put_nowait(detections)
                return
            except Full:
                time.sleep(0.01)

    def queue_confirmed_tracklet(self, confirmed_tracked):
        while self.seq_info["show_tracklets"]:
            try:
                self._tracker_queue.put_nowait(confirmed_tracked)
                return
            except Full:
                time.sleep(0.01)

    def queue_tracklets(self, tracks):
        confirmed_tracked = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            entry = {}
            entry["trackID"] = track.track_id
            entry["bboxs"] = track.to_tlwh()
            confirmed_tracked.append(entry)
            # self.viewer.color = create_unique_color_uchar(track.track_id)
            # self.viewer.rectangle(
            #     *track.to_tlwh().astype(np.int), label=str(track.track_id))

        while self.seq_info["show_tracklets"]:
            try:
                self._tracker_queue.put_nowait(confirmed_tracked)
                return
            except Exception:
                time.sleep(0.01)


    def _display_image(self):
        '''
        NOT a thread-safe so it should be call 1 time only even there
        is multiple sequence
        :return: None
        '''
        # is_cv_show = False
        self._displayState = ThreadState.Init
        while True:

            if self._display_queue.empty():
                if self._terminate:
                    self._displayState = ThreadState.Finished
                    break

                if self._loadingState == ThreadState.Finished:
                    if self._displayState == ThreadState.Running:
                        try:
                            self.image[:] = 0
                            cv2.destroyWindow(self.window_name)
                        except Exception as e:
                            print(e)
                        # print("Finished {}".format(self.window_name))
                        self._displayState = ThreadState.Finished
                    time.sleep(0.1)
                    continue

            t0 = time.time()
            try:
                self.image = self._display_queue.get_nowait()
            except Empty:
                time.sleep(0.01)
                continue

            while self.seq_info["show_detections"]:
                try:
                    detections = self._detection_queue.get_nowait()
                    self.thickness = 2
                    self.color = 0, 0, 255
                    for detection in detections:
                        self.rectangle(*detection.tlwh, label=str(int(detection.label)), pos=1)
                    self._detection_queue.task_done()
                    break
                except Empty:
                    time.sleep(0.001)
                    if self._terminate:
                        break

            while self.seq_info["show_tracklets"]:
                try:
                    tracks = self._tracker_queue.get_nowait()
                    # self.thickness = 2
                    for track in tracks:
                        self.color = create_unique_color_uchar(track["trackID"])
                        # bboxs = track.to_tlwh().astype(np.int)
                        ret = track["bboxs"]
                        # ret[2] *= ret[3]
                        # ret[:2] -= ret[2:] / 2
                        self.rectangle(*ret, label=str(track["trackID"]))
                    self._tracker_queue.task_done()
                    break
                except Empty:
                    time.sleep(0.001)
                    if self._terminate:
                        break

            resized_img = cv2.resize(self.image, self._image_shape[:2])
            cv2.imshow(self.window_name, resized_img)
            # is_cv_show = True
            t1 = time.time()
            remaining_time = max(1, int(self.seq_info["update_ms"] - 1e3 * (t1 - t0)))


            self._display_queue.task_done()
        print("Exit")

        try:
            self.image[:] = 0
            cv2.destroyWindow(self.window_name)
        except Exception as e:
            print(e)

    def stop(self):
        print("STOP VISUALIZER NOW")
        self._terminate = True
        self._loading_thread.join()
        self._display_thread.join()
        if self.vcap:
            self.vcap.release()

    def __del__(self):
        try:
            self.stop()
        except Exception as e:
            print(e)





if __name__ == "__main__":

    seq_info = {}
    seq_info["input_type"] = "video"
    seq_info["video_path"] =  0 #"http://192.168.0.22:8080/video"
    seq_info["update_ms"] = 20
    seq_info["show_detections"] = False
    seq_info["show_tracklets"] = False
    seq_info["image_shape"] = 480
    pool_display = PoolLoader(20)

    pool_display.load(seq_info)
    while True:
        try:
            pool_display.read()
        except KeyboardInterrupt:
            break

    pool_display.stop()

    # seq_info = {}
    # seq_info["input_type"] = "images"
    # seq_info["update_ms"] = 60
    # seq_info["show_detections"] = False
    # seq_info["show_tracklets"] = False
    # pool_display = PoolLoader(None)
    #
    # dataset = r"/datasets/kitti_tracking/image/"
    # for sequence in os.listdir(dataset):
    #     sequence_dir = os.path.join(dataset, sequence)
    #     seq_info["imgdir"] = sequence_dir
    #     seq_info["sequence_name"] = sequence
    #     pool_display.load(seq_info)
    #     while True:
    #         try:
    #             pool_display.read()
    #         except Exception as e:
    #             break
    #
    # pool_display.stop()