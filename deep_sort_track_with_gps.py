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
from visualizer.VideoLoader import VideoLoader, ImageLoader, NdImageLoader, ImageLoaderwithGPS
# --sequence_dir=/media/msis_dasol/1TB/dataset/test/MOT16-06 --detection_file=/media/msis_dasol/1TB/nn_pretrained/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100
from encoder.TripletNet import TripletNet
from encoder.PseudoEncoder import PseudoEncoder
import tensorflow as tf
from evaluator.Evaluator import Evaluator
# from embeddingIO.FeatureExchange import *
from tools.default_args import *
from mctracker.mctracker import MultiCameraTracker

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

    mctracker = MultiCameraTracker(args.mc_mode, args.bind_port, args.server_addr)
    evaluator = Evaluator()

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

    if running_cfg == "from_detect":
        print("THIS MODE WILL RUNNING FROM SCRATCH!!!!")
        print("Detector type {} config: {}".format(detection_cfg['type'], detection_cfg['name']))

        if detection_cfg['type'] == "Yolo":
            detector = Detector(configPath=detection_cfg['cfg_path'], metaPath=detection_cfg['meta_path'],
                                weightPath=detection_cfg['weight_path'], sharelibPath=detection_cfg['sharelibPath'],
                                gpu_num=args.gpu)
        else:
            detector = TensorFlowDetector(sess, detection_cfg['frozenpb'], class_filter=None,
                                          altName=detection_cfg['metaFile'])

    if running_cfg == "from_detect" or running_cfg == "from_encoded":
        print("Extractor config: %s" % extractor_cfg['name'])

        encoder = TripletNet(sess, extractor_cfg['frozen_ckpt'], detection_cfg['class_filter'], args.extractor_batchsize)

    # specific_sequence = args.sequence
    selected_sample = np.load(args.selected_sample)
    for sequence in selected_sample.astype(np.int64):
        if len(args.sequence) > 0 and str(sequence) != args.sequence:
            continue
        seqname = "{}.npy".format(sequence)
        sequence_path = os.path.join(args.gps_sample_dir, seqname)
        tmp = np.load(sequence_path)
        if tmp.shape[0] < args.minium_gps_len:
            continue
        if os.path.isfile(sequence_path):
            if running_cfg == "from_encoded":
                raw_detections_file = os.path.join(raw_detections_dir, "%s.npy" % sequence)
                detector = PseudoDetector(detFile=raw_detections_file, metaFile=detection_cfg['metaFile'])
            elif running_cfg == "track":
                detector = NoneDetector(metaFile=detection_cfg['metaFile'])
                detection_file = os.path.join(detections_dir, "%s.npy" % sequence)
                encoder = PseudoEncoder(detection_file, True)
            '''
                runway_1 = 15785476517
                runway_2 = 15785477680
                red_stop_1 = 15785479180
            '''
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", args.max_cosine_distance, args.nn_budget)
            tracker = Tracker(metric, encoder)
            mctracker.updateSingleTracker(tracker, metric)
            
            def frame_callback(vis, frame, frame_idx):
                print("\r Processing frame %05d" % frame_idx, end='')

                _t0 = time.time()
                raw_detections = detector(frame, frame_idx)
                _t1 = time.time() - _t0

                # preprocess the result first to reduce the embedding computation
                # Run non-maxima suppression.
                if running_cfg != "track":
                    detections = []
                    boxes = []
                    scores  = []
                    for idx, raw_detection in enumerate(raw_detections):
                        if raw_detection[detector.SCORE] >= args.min_confidence and raw_detection[detector.HEIGHT] > args.min_detection_height:
                            boxes.append([raw_detection[detector.TOP], raw_detection[detector.LEFT], raw_detection[detector.WIDTH], raw_detection[detector.HEIGHT]])
                            scores.append(raw_detection[detector.SCORE])
                            detections.append(raw_detection)

                    boxes = np.asarray(boxes)
                    indices = preprocessing.non_max_suppression(
                        boxes, args.nms_max_overlap, scores)

                    raw_detections = [detections[i] for i in indices]

                detections = encoder(frame, raw_detections, frame_idx)

                # Update tracker.
                tracker.predict()
                tracker.update(detections, frame, frame_idx)
                _t2 = time.time() - _t0 - _t1
                # print(seq_info["detections"][frame_idx][7])
                # Store results.
                # track_id list
                for track in tracker.tracks:
                    if not track.is_predicted() and not track.is_confirmed() or track.time_since_update > 0:
                        continue
                        # track.to_tlbr()
                    bbox = track.detection_bboxs.astype(np.int)
                    bbox[2:] += bbox[:2]
                    mctracker.initialize_ego_track(track)
                    mctracker.broadcast(encoder.update_trackid(track.detection_id, track.track_id))
                    class_id = encoder.get_class_id(track.detection_id)
                    class_name = detector.altNames[class_id]
                    evaluator.append(frame_idx, track.track_id, bbox, class_name)
                    # mctracker.client_Q.put()
                    # left top right bottom
                # mctracker.broadcastEmb()
                mctracker.filter_missing_track()
                # mctracker.sendAllPayload()

                try:
                    matching = mctracker.agrregate(frame_idx)
                except Exception as e:
                    print(e)
                    matching = []
                # Update visualization.
                if args.display:
                    vis.set_image(frame.copy())
                    vis.viewer.annotate(4, 20, "dfps {:03.1f} tfps {:03.1f}".format(1 / _t1, 1 / _t2))
                    vis.draw_detections(detections)
                    vis.draw_trackers_with_othertag(tracker.tracks, matching, False, mctracker.running_mode, args.debug_level)
                    vis.viewer.show_image()
                # notify other tracker or wait here

                mctracker.finished()
            # gps_sample = '/datasets/sample_video/2020_01_09/bwm_x5/image_left/gps/below_40m/15785475403.npy'
            # gps_sample = '/datasets/sample_video/2020_01_09/kia_niro/image_left/gps/below_40m/15785475403.npy'
            # Run tracker.
            if args.display:
                visualizer = ImageLoaderwithGPS(args.sequence_dir, 30, sequence_path, running_name, args.start_frame, args.crop_area)
                if args.save_video:
                    visualizer.viewer.enable_videowriter(os.path.join(video_dir, "%s.avi" % sequence), fps=5)
            else:
                visualizer = NdImageLoader(sequence_path)

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
    args.add_argument("--gps_sample_dir", help="sample gps dir", required=True)
    args.add_argument("--selected_sample", help="npy for selected sample", required=True)
    args.add_argument("--minium_gps_len", help="minimum gps sequence leng",
                      type=int, default=60)

    run(args.parse_args())
