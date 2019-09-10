# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import time

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detector.DarknetDetector import Detector
from detector.PseudoDetector import PseudoDetector, NoneDetector
# --sequence_dir=/media/msis_dasol/1TB/dataset/test/MOT16-06 --detection_file=/media/msis_dasol/1TB/nn_pretrained/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100
from encoder import extract_image_patch
from encoder.TripletNet import TripletNet
from encoder.PseudoEncoder import PseudoEncoder
import pathlib
import tensorflow as tf

def create_box_encoder(sess, model_filename, class_filter, batch_size=32):
    image_encoder = TripletNet(sess, model_filename)
    image_shape = image_encoder.image_shape

    def encoder(image, detect_res):
        image_patches = []
        image_patches_id = []
        for idx, detect in enumerate(detect_res):
            if detect[7] in class_filter:
                patch = extract_image_patch(image, detect[2:6], image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(detect[2:6]))
                    patch = np.random.uniform(
                        0., 255., image_shape).astype(np.uint8)
                image_patches_id.append(idx)
                image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        features = image_encoder(image_patches, batch_size)
        # return as MOT 16 format with detection
        # detections = [(detect_res[id][0], -1,
        #                detect_res[id][1][0], detect_res[id][1][1], detect_res[id][1][2], detect_res[id][1][3],
        #                detect_res[id][2], detect_res[id][3], -1, -1, features[idx])
        #               for idx, id in enumerate(image_patches_id)]
        detections = [np.r_[detect_res[id][0:10], features[idx]] for idx, id in enumerate(image_patches_id)]
        # detections = [Detection(detect_res[id][0], detect_res[id][1], features[idx], detect_res[id][3]) for idx, id in enumerate(image_patches_id)]
        return detections

    return encoder


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir)
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature, label = row[2:6], row[6], row[10:], row[7]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature, label))
    return detection_list

def create_detection_from_list(detection_mat, min_height=0):
    detection_list = []
    for row in detection_mat:
        bbox, confidence, feature, label = row[2:6], row[6], row[10:], row[7]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature, label))
    return detection_list


def run(dataset_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the Image dataset  directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    enable_video = True
    running_name = "msis_tracking_dataset_2"
    running_cfg = "from_encoded" # there will be 3 mode: run from from_detect, from_encoded or track
    running_cfg = "from_detect"
    # running_cfg = "track"
    # encoded_detection_file = os.path.join(detections_dir, sequence_name + ".npy")
    detection_file = None  # we set
    if running_cfg == "from_detect":
        print("THIS MODE WILL RUNNING FROM SCRATCH!!!!")
        cfg_path = "../alex_darknet/cfg/yolov3.cfg"
        meta_path = "detector/cfg/coco.data"
        weight_path = "../alex_darknet/yolov3.weights"
        detector = Detector(configPath=cfg_path, metaPath=meta_path, weightPath=weight_path,
                            sharelibPath="./libdarknet.so", gpu_num=0)
    if running_cfg == "from_detect" or running_cfg == "from_encoded":
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        config.gpu_options.visible_device_list = str(0)
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        frozen_ckpt = "/home/msis_member/Project/triplet-reid/experiment/VeRi/run3/encoder_trinet.pb"
        class_filter = [2, 5, 7]
        encoder = TripletNet(sess, frozen_ckpt, class_filter)


    # if (os.path.exists(encoded_detection_file)):
    specific_sequence = "video6"
    specific_sequence =""
    result_folder = os.path.join("results", running_name)
    raw_detections_dir = os.path.join(result_folder, "raw_detections")
    detections_dir = os.path.join(result_folder, "detections")
    track_dir = os.path.join(result_folder, "tracks")
    video_dir = os.path.join(result_folder, "video")
    os.makedirs(raw_detections_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    for sequence in os.listdir(dataset_dir):
        if len(specific_sequence) > 0 and sequence != specific_sequence:
            continue
        sequence_dir = os.path.join(dataset_dir, sequence)
        if os.path.isdir(sequence_dir):
            if running_cfg == "from_encoded":
                raw_detections_file = os.path.join(raw_detections_dir, "%s.npy" % sequence)
                detector = PseudoDetector(detFile=raw_detections_file)
            elif running_cfg == "track":
                detector = NoneDetector()
                detection_file = os.path.join(detections_dir, "%s.npy" % sequence)
                encoder = PseudoEncoder(detection_file, True)

            seq_info = gather_sequence_info(sequence_dir, None)

            metric = nn_matching.NearestNeighborDistanceMetric(
                "euclidean", max_cosine_distance, nn_budget)
            tracker = Tracker(metric)
            # results = []
            def frame_callback(vis, frame_idx):
                print("Processing frame %05d" % frame_idx)
                _t0 = time.time()
                image = cv2.imread(
                    seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

                _t0 = time.time()
                # detections = detector(seq_info["image_filenames"][frame_idx])
                raw_detections = detector(image, frame_idx)
                _t1 = time.time() - _t0
                # bbxs = [d[0] for d in detect_res if d[1] >= min_confidence and d[3] == "car" or d[3] == "truck" or d[3] == "bus"]
                detections = encoder(image, raw_detections, frame_idx)
                # np_detections =
                # detections = create_detections(np.asarray(detection_n_feature), frame_idx, min_detection_height)

                detections = [d for d in detections if d.confidence >= min_confidence]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                                        boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Update tracker.
                tracker.predict()
                tracker.update(detections)
                _t2 = time.time() - _t0 - _t1
                # Update visualization.
                if display:
                    vis.set_image(image.copy())
                    vis.viewer.annotate(4, 20, "dfps {:03.1f} tfps {:03.1f}".format(1/_t1, 1/_t2))
                    vis.draw_detections(detections)
                    vis.draw_trackers(tracker.tracks)
                # print(seq_info["detections"][frame_idx][7])
                # Store results.
                # for track in tracker.tracks:
                #     if not track.is_confirmed() or track.time_since_update > 1:
                #         continue
                #     bbox = track.to_tlbr()
                #     # left top right bottom
                #     results.append([
                #         frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

            # Run tracker.
            if display:
                visualizer = visualization.Visualization(seq_info, update_ms=5)
                visualizer.viewer.enable_videowriter(os.path.join(video_dir, "%s.avi" % sequence), fps=5)
            else:
                visualizer = visualization.NoVisualization(seq_info)
            try:
                visualizer.run(frame_callback)
            except Exception as e:
                print(e)
            finally:
                detector.save(raw_detections_dir, sequence)
                encoder.save(detections_dir, sequence)
                # raw_detections_np = np.asarray(raw_detections)
                # np.savetxt(os.join.path(raw_detections_dir, "") raw_detections_np, )
    if running_cfg == "from_detect" or running_cfg == "from_encoded":
        sess.close()
    # Store results.

    # f = open(output_file, 'w')
    # for row in results:
    #     # if not loading_groundtruth and eval_3d is True and(t_data.X==-1000 or t_data.Y==-1000 or t_data.Z==-1000):
    #     #
    #     # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #     #     row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    #     # KITTI_LABEL = ["frame", "track_id", "class_name", "truncated",
    #     #                "occluded", "alpha", "bbox_l", "bbox_t",
    #     #                "bbox_r", "bbox_b", "hdim", "wdim",
    #     #                "ldim", "locx", "locy", "locz", "rot_y"]
    #     if 0:
    #         # continue
    #         print('%d %d Pedestrian 0 0 0 %.2f %.2f %.2f %.2f 0 0 0 -1000 -1000 -1000 0' % (
    #             row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
    #     else:
    #         print('%d %d Car 0 0 0 %.2f %.2f %.2f %.2f 0 0 0 -1000 -1000 -1000 0' % (
    #             row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=r"/media/msis_dasol/1TB/dataset/test/MOT16-06", required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.",
        default=r"/media/msis_dasol/1TB/nn_pretrained/MOT16_POI_test",
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="./hypotheses_euclidean.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
