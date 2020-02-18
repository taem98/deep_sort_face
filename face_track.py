# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import time
import numpy as np
import json

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detector.DarknetDetector import Detector
from detector.TensorFlowDetector import TensorFlowDetector
from detector.PseudoDetector import PseudoDetector, NoneDetector
from visualizer.VideoLoader import VideoLoader, ImageLoader, NdImageLoader
# --sequence_dir=/media/msis_dasol/1TB/dataset/test/MOT16-06 --detection_file=/media/msis_dasol/1TB/nn_pretrained/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100
from encoder.TripletNet import TripletNet
from encoder.PseudoEncoder import PseudoEncoder
import tensorflow as tf
from evaluator.Evaluator import Evaluator
# from embeddingIO.FeatureExchange import *
from tools.default_args import *
from mctracker.mctracker import MultiCameraTracker
from visualizer.PoolLoader import PoolLoader

def load_json_config(jsonfile):
    with open(jsonfile, "r") as f:
        config = json.load(f)
    return config


def run(args):
    running_name = args.running_name
    running_cfg = args.running_cfg

    result_folder = os.path.join("results", running_name)
    raw_detections_dir = os.path.join(result_folder, "raw_detections")
    detections_dir = os.path.join(result_folder, "detections")
    track_dir = os.path.join(result_folder, "tracks")
    video_dir = os.path.join(result_folder, "video")
    os.makedirs(raw_detections_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    # dump all the setting to later debug
    with open(os.path.join(result_folder, "args.json"), 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    # mctracker = MultiCameraTracker(args.mc_mode, args.bind_port, args.server_addr)
    # evaluator = Evaluator()

    try:
        detection_cfg = load_json_config(os.path.join(result_folder, "config_detector.json"))
        extractor_cfg = load_json_config(os.path.join(result_folder, "config_extractor.json"))
        seq_info = load_json_config(os.path.join(result_folder, "config_loader.json"))
    except Exception as e:
        print(e)
        print("MUST CONFIGURE THE CORRECT config_detector and config_extractor file")
        return


    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.visible_device_list = str(args.gpu)
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)

    # if running_cfg == "from_detect":
    print("THIS MODE WILL RUNNING FROM SCRATCH!!!!")
    print("Detector type {} config: {}".format(detection_cfg['type'], detection_cfg['name']))

    if detection_cfg['type'] == "Yolo":
        detector = Detector(configPath=detection_cfg['cfg_path'], metaPath=detection_cfg['meta_path'],
                            weightPath=detection_cfg['weight_path'], sharelibPath=detection_cfg['sharelibPath'],
                            gpu_num = args.gpu)
    else:
        detector = TensorFlowDetector(sess, detection_cfg['frozenpb'], class_filter=None, altName=detection_cfg['metaFile'])

    encoder = TripletNet(sess, extractor_cfg['frozen_ckpt'], detection_cfg['class_filter'],
                         args.extractor_batchsize)
    detector.isSaveRes = False
    encoder.isSaveRes = False

    pool_display = PoolLoader(20)

    specific_sequence = args.sequence

    def run():
        for sequence in os.listdir(args.sequence_dir):
            if len(specific_sequence) > 0 and sequence != specific_sequence:
                continue
            sequence_dir = os.path.join(args.sequence_dir, sequence)
            if os.path.isdir(sequence_dir):

                seq_info["imgdir"] = sequence_dir
                seq_info["sequence_name"] = sequence
                pool_display.load(seq_info)
                metric = nn_matching.NearestNeighborDistanceMetric(
                    "cosine", args.max_cosine_distance, args.nn_budget)
                tracker = Tracker(metric, encoder, max_iou_distance=0.6, n_init=3)
                # mctracker.updateSingleTracker(tracker, metric)

                while True:
                    try:
                        frame_idx, frame = pool_display.read()
                    except Exception as e:
                        print(e)
                        break

                    # print("Processing frame %05d" % frame_idx)

                    _t0 = time.time()
                    raw_detections = detector(frame, frame_idx)

                    detections = []
                    boxes = []
                    scores = []
                    for idx, raw_detection in enumerate(raw_detections):
                        if raw_detection[detector.SCORE] >= args.min_confidence and raw_detection[detector.HEIGHT] > args.min_detection_height:
                            boxes.append([raw_detection[detector.TOP], raw_detection[detector.LEFT],
                                          raw_detection[detector.WIDTH], raw_detection[detector.HEIGHT]])
                            scores.append(raw_detection[detector.SCORE])
                            detections.append(raw_detection)

                    boxes = np.asarray(boxes)
                    indices = preprocessing.non_max_suppression(
                        boxes, args.nms_max_overlap, scores)

                    raw_detections = [detections[i] for i in indices]

                    # _t1 = time.time() - _t0
                    detections = encoder(frame, raw_detections, frame_idx)
                    # _t2 = time.time() - _t1

                    # Update tracker.
                    tracker.predict()
                    tracker.update(detections, frame, frame_idx)
                    pool_display.last_proctime = time.time() - _t0

                    pool_display.queue_detection(detections)

                    confirmed_tracked = []
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        entry = {}
                        entry["trackID"] = track.track_id
                        entry["bboxs"] = track.to_tlwh()
                        confirmed_tracked.append(entry)

                    pool_display.queue_confirmed_tracklet(confirmed_tracked)


    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise(e)
    finally:
        print("Release")
        pool_display.stop()
        sess.close()

if __name__ == "__main__":
    args = parse_args()
    run(args.parse_args())
