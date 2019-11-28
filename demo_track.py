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
    print("Detector config: %s" % detection_cfg['name'])

    frozenpb = r"/home/msis_member/.keras/datasets/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.pb"
    metaPath = r"/home/msis_member/Project/deep_sort/detector/data/kitti_tensorflow.names"
    detector = TensorFlowDetector(sess, frozenpb, class_filter=None, altName=metaPath)
    encoder = TripletNet(sess, extractor_cfg['frozen_ckpt'], detection_cfg['class_filter'],
                         args.extractor_batchsize)
    # if running_cfg == "from_detect" or running_cfg == "from_encoded":
        # print("Extractor config: %s" % extractor_cfg['name'])
        #

    specific_sequence = args.sequence

    for sequence in os.listdir(args.sequence_dir):
        if len(specific_sequence) > 0 and sequence != specific_sequence:
            continue
        sequence_dir = os.path.join(args.sequence_dir, sequence)
        if os.path.isdir(sequence_dir):
            # if running_cfg == "from_encoded":
            #     raw_detections_file = os.path.join(raw_detections_dir, "%s.npy" % sequence)
            #     detector = PseudoDetector(detFile=raw_detections_file, metaFile=detection_cfg['metaFile'])
            # elif running_cfg == "track":
            #     detector = NoneDetector(metaFile=detection_cfg['metaFile'])
            #     detection_file = os.path.join(detections_dir, "%s.npy" % sequence)
            #     encoder = PseudoEncoder(detection_file, True)

            # metric = nn_matching.NearestNeighborDistanceMetric(
            #     "cosine", args.max_cosine_distance, args.nn_budget)
            # tracker = Tracker(metric, max_iou_distance=0.6)
            # mctracker.updateSingleTracker(tracker, metric)

            seq_info = {}
            seq_info["imgdir"] = sequence_dir
            seq_info["update_ms"] = 30
            seq_info["sequence_name"] = specific_sequence
            pool_display = PoolLoader(seq_info, None)

            while True:
                try:
                    frame_idx, frame = pool_display.read()
                except Exception as e:
                    print(e)
                    break

                print("Processing frame %05d" % frame_idx)

                _t0 = time.time()
                raw_detections = detector(frame, frame_idx)
                _t1 = time.time() - _t0
                detections = encoder(frame, raw_detections, frame_idx)
                pool_display.queue_detection(detections)

            pool_display.stop()

if __name__ == "__main__":
    args = parse_args()
    run(args.parse_args())
