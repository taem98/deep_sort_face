import numpy as np
import os
import mat4py
from natsort import natsorted

def generate_from_matfile():
    file_path = "../tmp/rrc/RRC_Detections_mat"
    result_dir = '../results'
    datasets = os.listdir(file_path)
    print(datasets)
    for dataset in datasets:
        subset = os.path.join(file_path, dataset)
        subset_res = os.path.join(result_dir, "kitti_rrc_{}_002".format(dataset))
        os.makedirs(subset_res, exist_ok=True)
        subset_det_dir = os.path.join(subset_res, "raw_detections")
        os.makedirs(subset_det_dir, exist_ok=True)
        for sequence in os.listdir(subset):
            sequence_dir = os.path.join(subset, sequence)
            file = [f for f in os.listdir(sequence_dir) if "detections" in f]
            dets = []
            detections = mat4py.loadmat(os.path.join(sequence_dir, file[0]))
            detections = detections['detections']
            for idx, det in enumerate(detections):
                for l0 in det:
                    if isinstance(l0, list):
                        if len(l0) > 0:
                            if isinstance(l0[0], list):
                                for l in l0:
                                    dets.append([idx, -1, l[0], l[1], l[2] - l[0], l[3] - l[1], l[4], 0, -1, -1])
                            else:
                                l = l0
                                dets.append([idx, -1, l[0], l[1], l[2] - l[0], l[3] - l[1], l[4], 0, -1, -1])
            dets_np = np.asarray(dets)
            # seq_name =
            np.save(os.path.join(subset_det_dir, "{}.npy".format(sequence)), dets_np, allow_pickle=False)
            np.savetxt(os.path.join(subset_det_dir, "{}.txt".format(sequence)), dets_np, fmt='%4.2f')
            # print(detections)


def generate_from_det_file(file_path, result_name):
    result_dir = '../results'
    subset_res = os.path.join(result_dir, result_name)
    os.makedirs(subset_res, exist_ok=True)
    subset_det_dir = os.path.join(subset_res, "raw_detections")
    os.makedirs(subset_det_dir, exist_ok=True)
    for sequence in os.listdir(file_path):
        sequence_dir = os.path.join(file_path, sequence)
        files = natsorted(os.listdir(sequence_dir))
        dets = []
        for idx, f in enumerate(files):
            print("\r sequence {} file {}".format(sequence, f), end='\t\t')
            with open(os.path.join(sequence_dir, f), 'r') as fread:
                fname = os.path.splitext(f)[0]
                for line in fread.readlines():
                    frame_idx = int(fname)
                    if frame_idx != idx:
                        print("ID mismatch {} frame_id {}: idx {}".format(f, frame_idx, idx))
                    line = line.rstrip().replace(',', '')
                    res = [float(split) for split in line.split(' ')]
                    res.insert(0, float(frame_idx))
                    dets.append(res)
        dets_np = np.asarray(dets)
        # seq_name =
        np.save(os.path.join(subset_det_dir, "{}.npy".format(sequence)), dets_np, allow_pickle=False)
        np.savetxt(os.path.join(subset_det_dir, "{}.txt".format(sequence)), dets_np, fmt='%4.2f')

if __name__ == "__main__":
    generate_from_det_file("/home/msis_member//Project/rrc_detection/results/image/", "kitti_train_rrc_002")
    generate_from_det_file("/home/msis_member//Project/rrc_detection/results/image_03/", "kitti_train_rrc_003")
    generate_from_det_file("/home/msis_member//Project/rrc_detection/results/image_test_02/", "kitti_test_rrc_002")
    generate_from_det_file("/home/msis_member//Project/rrc_detection/results/image_test_03/", "kitti_test_rrc_003")