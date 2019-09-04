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
from darknet.Detector import Detector
# --sequence_dir=/media/msis_dasol/1TB/dataset/test/MOT16-06 --detection_file=/media/msis_dasol/1TB/nn_pretrained/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100
from tripletnet_encoder import extract_image_patch
from tripletnet_encoder.TripletNet import TripletNet

def create_box_encoder(model_filename, class_filter, gpu_num=0, batch_size=32):
    image_encoder = TripletNet(model_filename, gpu_num=gpu_num)
    image_shape = image_encoder.image_shape

    def encoder(image, detect_res):
        image_patches = []
        image_patches_id = []
        for idx, detect in enumerate(detect_res):
            if detect[3] in class_filter:
                patch = extract_image_patch(image, detect[0], image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(box))
                    patch = np.random.uniform(
                        0., 255., image_shape).astype(np.uint8)
                image_patches_id.append(idx)
                image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        features = image_encoder(image_patches, batch_size)
        detections = [Detection(detect_res[id][0], detect_res[id][1], features[idx], detect_res[id][3]) for idx, id in enumerate(image_patches_id)]
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
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
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
    cfg_path = "../alex_darknet/cfg/yolov3.cfg"
    meta_path = "darknet/cfg/coco.data"
    weight_path = "../alex_darknet/yolov3.weights"
    detector = Detector(configPath=cfg_path, metaPath=meta_path, weightPath=weight_path, sharelibPath="./libdarknet.so", gpu_num=0)
    encoder = create_box_encoder("/home/msis_member/Project/triplet-reid/experiment/VeRi/run3/encoder_trinet.pb", class_filter=['car', 'truck', 'bus'])
    detection_file = None # we set
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "euclidean", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    # results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)
        _t0 = time.time()
        image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        # Load image and generate detections.
        if seq_info["detections"] is not None:
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height)
        else:
        #     we generate the detection file here
            _t0 = time.time()
            # detections = detector(seq_info["image_filenames"][frame_idx])
            detect_res = detector(image, thresh=0.5)
            _t1 = time.time() - _t0
            # bbxs = [d[0] for d in detect_res if d[1] >= min_confidence and d[3] == "car" or d[3] == "truck" or d[3] == "bus"]
            detections = encoder(image, detect_res)

        detections = [d for d in detections if d.confidence >= min_confidence and d.tlwh[3] > min_detection_height]

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
        visualizer.viewer.enable_videowriter("detection_with_track.avi", fps=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

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
