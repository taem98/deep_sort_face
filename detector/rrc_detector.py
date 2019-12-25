import numpy as np
import os
import mat4py

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
