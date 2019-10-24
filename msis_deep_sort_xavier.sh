#!/usr/bin/env bash

python deep_sort_track.py --sequence_dir ../../datasets/ \
                          --running_name=slave_msis_7sec \
                          --extractor_batchsize 16 \
                          --gpu=0 \
                          --running_cfg=from_detect \
                          --sequence="frames_32_7Sec" \
                          --mc_mode=0 \
                          --crop_area 150 200 100 100 \
                          --start_frame=0