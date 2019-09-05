import numpy as np
import os


class NoneDetector(object):
    def __init__(self):
        pass

    def __call__(self, img, frame_idx):
        pass

    def save(self, output_dir, sequence):
        pass


class PseudoDetector(NoneDetector):
    def __init__(self, fromFile, detFile):
        super().__init__()
        self._from_file = True
        self._detections_list = []
        if not self._from_file:
            if not os.path.isfile(detFile):
                raise FileNotFoundError("Can not found detection file")
            self._raw_detection = np.load(detFile)
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
            np_arr = np.asarray(self._detections_list)
            output_filename = os.path.join(output_dir, "%s.npy" % sequence)
            output_txtname = os.path.join(output_dir, "%s.txt" % sequence)

            np.save(output_filename, np_arr, allow_pickle=False)
            np.savetxt(output_txtname, np_arr, delimiter='')