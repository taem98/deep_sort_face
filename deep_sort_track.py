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


def run(args):

    running_name = args.running_name
    running_cfg = args.running_cfg
    mctracker = MultiCameraTracker(args.mc_mode, args.bind_port, args.server_addr)
    evaluator = Evaluator()

    metaFile = "./detector/data/kitti.names"

    if running_cfg == "from_detect":
        print("THIS MODE WILL RUNNING FROM SCRATCH!!!!")
        cfg_path = "../alex_darknet/cfg/yolov3.cfg"
        meta_path = "detector/cfg/coco.data"
        weight_path = "../alex_darknet/yolov3.weights"
        detector = Detector(configPath=cfg_path, metaPath=meta_path, weightPath=weight_path,
                            sharelibPath="./libdarknet.so", gpu_num=args.gpu)
    if running_cfg == "from_detect" or running_cfg == "from_encoded":
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        config.gpu_options.visible_device_list = str(args.gpu)
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        frozen_ckpt = "./encoder_trinet.pb"
        # class_filter = [0, 2, 6, 7]
        class_filter = [2, 5, 7]
        encoder = TripletNet(sess, frozen_ckpt, class_filter)

    specific_sequence = args.sequence

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

    for sequence in os.listdir(args.sequence_dir):
        if len(specific_sequence) > 0 and sequence != specific_sequence:
            continue
        sequence_dir = os.path.join(args.sequence_dir, sequence)
        if os.path.isdir(sequence_dir):
            if running_cfg == "from_encoded":
                raw_detections_file = os.path.join(raw_detections_dir, "%s.npy" % sequence)
                detector = PseudoDetector(detFile=raw_detections_file, metaFile=metaFile)
            elif running_cfg == "track":
                detector = NoneDetector(metaFile=metaFile)
                detection_file = os.path.join(detections_dir, "%s.npy" % sequence)
                encoder = PseudoEncoder(detection_file, True)

            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", args.max_cosine_distance, args.nn_budget)
            tracker = Tracker(metric)
            mctracker.updateSingleTracker(tracker, metric)
            def frame_callback(vis, frame, frame_idx):
                print("Processing frame %05d" % frame_idx)

                _t0 = time.time()
                raw_detections = detector(frame, frame_idx)
                _t1 = time.time() - _t0

                detections = encoder(frame, raw_detections, frame_idx)

                detections = [d for d in detections if d.confidence >= args.min_confidence]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                                        boxes, args.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Update tracker.
                tracker.predict()
                tracker.update(detections)
                _t2 = time.time() - _t0 - _t1
                # print(seq_info["detections"][frame_idx][7])
                # Store results.
                # track_id list
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    mctracker.broadcast(encoder.update_trackid(track.detection_id, track.track_id))
                    class_id = encoder.get_class_id(track.detection_id)
                    class_name = detector.altNames[class_id]
                    evaluator.append(frame_idx, track.track_id, bbox, class_name)
                    # mctracker.client_Q.put()
                    # left top right bottom
                # mctracker.broadcastEmb()
                matching = mctracker.agrregate(frame_idx)

                # Update visualization.
                if args.display:
                    vis.set_image(frame.copy())
                    vis.viewer.annotate(4, 20, "dfps {:03.1f} tfps {:03.1f}".format(1 / _t1, 1 / _t2))
                    vis.draw_detections(detections)
                    vis.draw_trackers_with_othertag(tracker.tracks, matching, False)
                    vis.viewer.show_image()
                # notify other tracker or wait here
                mctracker.finished()

            # Run tracker.
            if args.display:
                visualizer = ImageLoader(sequence_dir, 5, running_name)
                if args.save_video:
                    visualizer.viewer.enable_videowriter(os.path.join(video_dir, "%s.avi" % sequence), fps=5)
            else:
                visualizer = NdImageLoader(sequence_dir)

            try:
                visualizer.run(frame_callback)
                mctracker.removeSingleTracker()
            except Exception as e:
                raise Exception("exception")
                print(e)

            finally:
                detector.save(raw_detections_dir, sequence)
                encoder.save(detections_dir, sequence)
                evaluator.save(result_folder, sequence)
                # server.stop(1)
                # raw_detections_np = np.asarray(raw_detections)
                # np.savetxt(os.join.path(raw_detections_dir, "") raw_detections_np, )
    # Store results.


if __name__ == "__main__":
    args = parse_args()
    run(args.parse_args())
