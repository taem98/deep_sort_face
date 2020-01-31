# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer


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


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        try:
            image_shape = seq_info['image_shape'], int(seq_info['image_shape'] * aspect_ratio)
        except KeyError:
            image_shape = 1024, int(aspect_ratio * 1024)

        import socket

        self.viewer = ImageViewer(
            update_ms, image_shape, "%s Figure %s" % (socket.gethostname(), seq_info["sequence_name"]))
        self.viewer.thickness = 2
        self.viewer.text_size = seq_info['text_size']
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image
        self.viewer.is_frame_updated = True

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh, label=str(int(detection.label)), pos=1)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#
    def draw_trackers_with_othertag(self, tracks, matching, detection_bbox = False, target_color=0, debug=0, evaluator=None):
        self.viewer.thickness = 2
        if debug == 2:
            for track in tracks:
                label_str = "{}".format(track.track_id)
                if not track.is_confirmed() or track.time_since_update > 0:
                    label_str += "_U:{}".format(track.time_since_update)

                label_str += "_{0:.2f}".format(track.affinity_score)
                label_str += "_{0:.2f}".format(track.iou_score)
                track.affinity_score = 0.0

                target_id = None
                for idx, t in enumerate(matching):
                    if t[0] == track.track_id:
                        label_str += ":{}".format(t[1])
                        target_id = t[1]
                        break
                if target_color > 1 and target_id:
                    self.viewer.color = create_unique_color_uchar(target_id)
                else:
                    self.viewer.color = create_unique_color_uchar(track.track_id)
                # self.viewer.rectangle(
                #     *track.to_tlwh().astype(np.int), label=label_str)
                if not detection_bbox:
                    self.viewer.rectangle(
                        *track.to_tlwh().astype(np.int), label=label_str)
                else:
                    self.viewer.rectangle(
                        *track.detection_bboxs.astype(np.int), label=label_str)
                if evaluator:
                    evaluator.append(self.frame_idx, )

        elif debug == 1:
            for track in tracks:
                if not track.is_predicted() and not track.is_confirmed() or track.time_since_update > 1:
                    continue
                label_str = "{}".format(track.track_id)
                if track.affinity_score > 0.1:
                    label_str += "_{0:.2f}".format(track.affinity_score)
                    label_str += "_{0:.2f}".format(track.iou_score)
                track.affinity_score = 0.0
                track.iou_score = 0.0
                target_id = None
                for idx, t in enumerate(matching):
                    if t[0] == track.track_id:
                        label_str += ":{}".format(t[1])
                        target_id = t[1]
                        break
                if target_color > 1 and target_id:
                    self.viewer.color = create_unique_color_uchar(target_id)
                else:
                    self.viewer.color = create_unique_color_uchar(track.track_id)


                # class_id = int(track.det_meta[-1][2])

                if not detection_bbox:
                    self.viewer.rectangle(
                        *track.to_tlwh().astype(np.int), label=label_str)
                else:
                    bbox = track.det_meta[-1][1].astype(np.int)
                    # bbox[2:] += bbox[:2]
                    self.viewer.rectangle(
                        *bbox, label=label_str)


        else:
            for track in tracks:
                if not track.is_predicted() and not track.is_confirmed() or track.time_since_update > 1:
                    continue
                label_str = "{}".format(track.track_id)
                target_id = None
                for idx, t in enumerate(matching):
                    if t[0] == track.track_id:
                        label_str += ":{}".format(t[1])
                        target_id = t[1]
                        break
                if target_color > 1 and target_id:
                    self.viewer.color = create_unique_color_uchar(target_id)
                else:
                    self.viewer.color = create_unique_color_uchar(track.track_id)
                # self.viewer.rectangle(
                #     *track.to_tlwh().astype(np.int), label=label_str)
                if not detection_bbox:
                    self.viewer.rectangle(
                        *track.to_tlwh().astype(np.int), label=label_str)
                else:
                    self.viewer.rectangle(
                        *track.detection_bboxs.astype(np.int), label=label_str)

                if evaluator and target_id:
                    bbox = track.detection_bboxs.astype(np.int)
                    bbox[2:] += bbox[:2]
                    evaluator.append2(target_id, bbox, int(track.class_labelid))