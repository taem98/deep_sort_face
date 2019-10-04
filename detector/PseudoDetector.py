import numpy as np
import os
import csv

class NoneDetector(object):
    def __init__(self, metaFile):
        self.altNames = None
        try:
            if os.path.exists(metaFile):
                with open(metaFile) as namesFH:
                    namesList = namesFH.read().strip().split("\n")
                    self.altNames = [x.strip() for x in namesList]
        except TypeError:
            pass
        pass

    def __call__(self, img, frame_idx):
        pass

    def save(self, output_dir, sequence):
        pass


class PseudoDetector(NoneDetector):
    def __init__(self, detFile, metaFile, fromFile=True):
        super().__init__(metaFile)
        self._from_file = fromFile
        self._detections_list = []
        # self._dtype = [("frame_id", np.int) ,("track_id", np.int), ("x", np.float32),
        #                ("y",np.float32), ("w", np.float32), ("h", np.float32),
        #                ("confidence", np.float32), ("classid", np.int), ("ud0", np.int), ("ud1", np.int), ("name", '<U5')]

        # self._dtype = [np.int, np.int, np.float32, np.float32,  np.float32, np.float32, np.float32, np.int,np.int, np.int, '<U5']

        if self._from_file:
            if not os.path.isfile(detFile):
                raise FileNotFoundError("Can not found detection file")
            self._raw_detection = np.load(detFile)
            # print(self._raw_detection[0,"frame_id"])
            self._frame_indices = self._raw_detection[:, 0].astype(np.int)

        # min_frame_idx = frame_indices.astype(np.int).min()
        # max_frame_idx = frame_indices.astype(np.int).max()

    def __call__(self, img, frame_idx):
        rows = None
        if self._from_file:
            mask = self._frame_indices == frame_idx
            rows = self._raw_detection[mask]
        return rows

    def save(self, output_dir, sequence):
        if not self._from_file:
            # detections = self._detections_list[0:9]
            # np_arr = np.asarray(self._detections_list, dtype=self._dtype)
            np_arr = np.asarray(self._detections_list, dtype=np.float32)
            output_filename = os.path.join(output_dir, "%s.npy" % sequence)
            output_txtname = os.path.join(output_dir, "%s.txt" % sequence)

            np.save(output_filename, np_arr, allow_pickle=False)
            np.savetxt(output_txtname, np_arr, fmt='%4.2f')
            # with open(output_txtname, 'w', newline='') as f:
            #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            #     wr.writerow(self._detections_list)
            self._detections_list = []