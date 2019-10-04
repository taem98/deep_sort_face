import os
import numpy as np

from deep_sort.detection import Detection

class PseudoEncoder(object):
    def __init__(self, detFile, fromFile, altName=None):
        self._from_file = fromFile
        self._detections_list = []
        self._altName=altName
        self._detection_id = 0
        if self._from_file:
            if not os.path.isfile(detFile):
                raise FileNotFoundError("Can not found detection file")
            self._raw_detection = np.load(detFile)
            # print(self._raw_detection[0,"frame_id"])
            self._frame_indices = self._raw_detection[:, 0].astype(np.int)

    # def get_detection(self, idx, trackid):
    #     if self._from_file:
    #         return self._raw_detection[idx]
    #     else:
    #         return  self._detections_list[idx]
    #
    def get_detections(self, ):
        if self._from_file:
            return self._raw_detection
        else:
            return self._detections_list

    def get_class_id(self, frameid):
        if self._from_file:
            return self._raw_detection[frameid][7].astype(np.int)
        else:
            return self._detections_list[frameid][7].astype(np.int)

    def update_trackid(self, frameid, trackid):
        if self._from_file:
            self._raw_detection[frameid][1] = trackid
            return self._raw_detection[frameid]
        else:
            self._detections_list[frameid][1] = trackid
            return self._detections_list[frameid]
    
    def __call__(self, image, raw_detections, frame_id):
        '''
        in here we need to mark the detection with index to retrieve the confirm track
        since tracker may remove the feature after the track is confirm
        but we might need to change it later
        '''
        res = []
        if self._from_file:
            mask = self._frame_indices == frame_id
            rows = self._raw_detection[mask]
        else:
            rows = raw_detections
            self._detections_list.extend(rows)

        for row in rows:
            if self._altName:
                bbox, confidence, feature, label = row[2:6], row[6], row[10:], self._altName[row[7]]
            else:
                bbox, confidence, feature, label = row[2:6], row[6], row[10:], row[7]
            res.append(Detection(bbox, confidence, feature, label, self._detection_id))
            self._detection_id += 1
        return res

    def save(self, output_dir, sequence):
        if not self._from_file:
            # detections = self._detections_list[0:9]
            # np_arr = np.asarray(self._detections_list, dtype=self._dtype)
            np_arr = np.asarray(self._detections_list, dtype=np.float32)
            output_filename = os.path.join(output_dir, "%s.npy" % sequence)
            # output_txtname = os.path.join(output_dir, "%s.txt" % sequence)
            np.save(output_filename, np_arr, allow_pickle=False)
            # np.savetxt(output_txtname, np_arr, fmt='%4.2f')
            # with open(output_txtname, 'w', newline='') as f:
            #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            #     wr.writerow(self._detections_list)
            self._detections_list = []
            self._detection_id = 0