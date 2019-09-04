#!/usr/bin/env bash

python evaluate_kitti.py  \
            --kitti_dir /media/msis_dasol/1TB/dataset/kitti_tracking/image \
            --detection_dir /media/msis_dasol/1TB/github/mywork/vehicle_identification/tmp/detections/keras_mobilenetv2_no_top \
            --output_dir /media/msis_dasol/1TB/github/mywork/devkit/python/results/keras_mobilenetv2_no_top/data \
            --min_detection_height 25 \
            --min_confidence=0.3 \
            --nn_budget=100