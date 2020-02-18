import tensorflow as tf
from detector import *
from detector.PseudoDetector import PseudoDetector
import numpy as np

from mtcnn import MTCNN
from keras import backend as K

class MtCNNDetector(PseudoDetector):

    def __init__(self, sess, class_filter, batch_size=1, altName=None):
        super().__init__(None, altName, False)
        self.session = sess
        K.set_session(self.session)
        self._detector = MTCNN()

    def __call__(self, img, frameid):
        out_put = self._detector.detect_faces(img)
        # {'box': [811, 235, 36, 47], 'confidence': 0.9863850474357605, '
        # keypoints': {'left_eye': (824, 253), 'right_eye': (842, 252), 'nose': (834, 264), 'mouth_left': (825, 273), 'mouth_right': (839, 273)}}
        # for out in out_put:
        #     print(out_put)
        res = [(frameid, -1, output_dict['box'][0], output_dict['box'][1], output_dict['box'][2], output_dict['box'][3], output_dict['confidence'], 0, -1, -1) for output_dict in out_put]

        if self.isSaveRes:
            self._detections_list.extend(res)
        return res