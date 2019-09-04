#!/usr/bin/env bash

python evaluate_kitti.py  \
            --kitti_dir ./datasets/kitti_tracking/image \
            --detection_dir ./tmp/detections/tripletnet_veri \
            --output_dir ./results/tripletnet_veri/data \
            --min_detection_height 25 \
            --min_confidence=0.3 \
            --nn_budget=100