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
from encoder.TripletNet import TripletNet, FaceNet
from encoder.PseudoEncoder import PseudoEncoder
import tensorflow as tf
from evaluator.Evaluator import Evaluator
# from embeddingIO.FeatureExchange import *
from tools.default_args import *
from mctracker.mctracker import MultiCameraTracker
from deep_sort.tracker import Tracker
from deep_sort.tracker import Track
from detector.MtCNN import MtCNNDetector

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
        vis_cfg = load_json_config(os.path.join(result_folder, 'config_visualizer.json'))
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

    detector = MtCNNDetector(sess, class_filter=None, altName=detection_cfg['metaFile'])

    encoder = FaceNet(sess, extractor_cfg['frozen_ckpt'], detection_cfg['class_filter'],
                      args.extractor_batchsize)
    detector.isSaveRes = False
    encoder.isSaveRes = False



    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", args.max_cosine_distance, args.nn_budget)
    tracker = Tracker(metric, encoder)
    # mctracker.updateSingleTracker(tracker, metric)

    rg_name_data = []
    rg_feature_data = []
    try:
        all_data = np.load("./registed_data.npz", allow_pickle=True)
        rg_name_data = all_data['name']
        rg_name_data = rg_name_data.tolist()
        rg_feature_data = all_data['feature']
        rg_feature_data = rg_feature_data.tolist()
        all_data.close()
        print(rg_name_data)
        if all_data:
            pass
        else:
            rg_name_data = []
            rg_feature_data = []
        print("load data")
    except Exception as ex:
        print(ex)

    for idx in range(len(rg_name_data)):
        tracker.tracks.append(Track(np.array([]), np.array([]), tracker._next_id, tracker.n_init, tracker.max_age, None))
        tracker.tracks[idx].set_name_feature(rg_name_data[idx], rg_feature_data[idx])
        tracker.tracks[idx].set_registed_status()
        tracker._next_id += 1

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
            scores = []
            for idx, raw_detection in enumerate(raw_detections):
                if raw_detection[detector.SCORE] >= args.min_confidence and raw_detection[
                    detector.HEIGHT] > args.min_detection_height:
                    boxes.append([raw_detection[detector.TOP], raw_detection[detector.LEFT],
                                  raw_detection[detector.WIDTH], raw_detection[detector.HEIGHT]])
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
            bbox = track.det_meta[-1][1].astype(np.int)
            bbox[2:] += bbox[:2]
            class_id = int(track.det_meta[-1][2])
            mctracker.initialize_ego_track(track, frame.shape, frame_idx)
            class_name = detector.altNames[class_id]
            evaluator.append(frame_idx, track.track_id, bbox, class_name)

        if args.display:

            vis.set_image(frame.copy())
            vis.viewer.annotate(4, 20, "dfps {:03.1f} tfps {:03.1f}".format(1 / _t1, 1 / _t2))
            # vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            # vis.draw_trackers_with_othertag(tracker.tracks, matching, True, mctracker.running_mode, args.debug_level)
            key = vis.viewer.show_image()
            if key ==ord("r"):
                try:
                    number, name = input("number name : ").split()
                    for track in tracker.tracks:
                        if track.track_id == int(number):
                            track.name = name
                            rg_name_data.append(name)
                            rg_feature_data.append(tracker.metric.samples[track.track_id])
                        else:
                            pass
                except:
                    pass
            if key ==ord("d"):
                try:
                    name = input("name : ")

                    del_idx = rg_name_data.index(name)
                    del rg_name_data[del_idx]
                    del rg_feature_data[del_idx]

                    for track in tracker.tracks:
                        if track.name == name:
                            track.name = ""
                            track.state = 3 # deleted
                            break
                        else:
                            pass
                except Exception as ex:
                    print(ex)
    # Run tracker.

    vis_cfg['videopath'] = 0
    visualizer = VideoLoader(vis_cfg)
    # if args.save_video:
    #     visualizer.viewer.enable_videowriter(os.path.join(video_dir, "%s.avi" % file_name), fps=5)


    try:
        visualizer.run(frame_callback)
        mctracker.removeSingleTracker()
    except Exception as e:
        raise Exception("exception")
        print(e)

    finally:
        print("registered")
        print(rg_name_data)
        np.savez_compressed("registed_data", name=rg_name_data, feature=rg_feature_data)

        # detector.save(raw_detections_dir, file_name)
        # encoder.save(detections_dir, file_name)
        # evaluator.save(result_folder, file_name)
            # server.stop(1)
            # raw_detections_np = np.asarray(raw_detections)
            # np.savetxt(os.join.path(raw_detections_dir, "") raw_detections_np, )
    # Store results.


if __name__ == "__main__":
    args = parse_args()
    run(args.parse_args())
