# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import time
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detector.DarknetDetector import Detector
from detector.PseudoDetector import PseudoDetector, NoneDetector
from visualizer.VideoLoader import VideoLoader, ImageLoader
# --sequence_dir=/media/msis_dasol/1TB/dataset/test/MOT16-06 --detection_file=/media/msis_dasol/1TB/nn_pretrained/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100
from encoder.TripletNet import TripletNet
from encoder.PseudoEncoder import PseudoEncoder
import tensorflow as tf
from evaluator.Evaluator import Evaluator
from embeddingIO.FeatureExchange import *

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
    running_name = "msis_tracking_dataset"
    running_cfg = "from_encoded" # there will be 3 mode: run from from_detect, from_encoded or track
    running_cfg = "from_detect"
    running_cfg = "track"
    # encoded_detection_file = os.path.join(detections_dir, sequence_name + ".npy")
    evaluator = Evaluator()
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
        frozen_ckpt = "./encoder_trinet.pb"
        class_filter = [2, 5, 7]
        encoder = TripletNet(sess, frozen_ckpt, class_filter)

    # if (os.path.exists(encoded_detection_file)):
    specific_sequence = "video6"
    # specific_sequence =""
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

            metric = nn_matching.NearestNeighborDistanceMetric(
                "euclidean", max_cosine_distance, nn_budget)
            tracker = Tracker(metric)

            # init the features exchange server

            # class
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
            embserver = EmbeddingsServing(tracker.tracks, encoder.get_detections())
            add_EmbServerServicer_to_server(embserver, server)
            # .add_GreeterServicer_to_server(self.embserver, self.server)
            server.add_insecure_port('[::]:50051')
            server.start()
            # results = []
            # start the server
            def frame_callback(vis, frame, frame_idx):
                print("Processing frame %05d" % frame_idx)

                _t0 = time.time()
                raw_detections = detector(frame, frame_idx)
                _t1 = time.time() - _t0

                detections = encoder(frame, raw_detections, frame_idx)

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
                    vis.set_image(frame.copy())
                    vis.viewer.annotate(4, 20, "dfps {:03.1f} tfps {:03.1f}".format(1/_t1, 1/_t2))
                    vis.draw_detections(detections)
                    vis.draw_trackers(tracker.tracks)
                # print(seq_info["detections"][frame_idx][7])
                # Store results.
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    evaluator.append(frame_idx, track.track_id, bbox, "Car")
                    # left top right bottom

            # Run tracker.
            if display:
                visualizer = ImageLoader(sequence_dir, 5)
                # visualizer.viewer.enable_videowriter(os.path.join(video_dir, "%s.avi" % sequence), fps=5)
            else:
                visualizer = visualization.NoVisualization(None)

            try:
                visualizer.run(frame_callback)
            except Exception as e:
                print(e)
            finally:
                detector.save(raw_detections_dir, sequence)
                encoder.save(detections_dir, sequence)
                evaluator.save(result_folder, sequence)
                server.stop(1)
                # raw_detections_np = np.asarray(raw_detections)
                # np.savetxt(os.join.path(raw_detections_dir, "") raw_detections_np, )
    if running_cfg == "from_detect" or running_cfg == "from_encoded":
        sess.close()
    # Store results.

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
