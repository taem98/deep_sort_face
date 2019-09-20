import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np
from deep_sort import linear_assignment
from deep_sort.detection import Detection

def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            linear_assignment.min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

# implement the tracker based on the previous running example from difference camera
# we must change to socket connection later
class MultiCameraTracker:
    def __init__(self, detection_file, metric, single_tracker):
        self._single_tracker = single_tracker
        self._othercamera_detection = np.load(detection_file)
        self.metric = metric

    def agrregate(self, frame_id):

        def distance_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i,10:] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return cost_matrix
        # we get the 2 previous frame from other camera
        _other_camera_indices = self._othercamera_detection[:, 0].astype(np.int)
        _other_track_indices = self._othercamera_detection[:, 1].astype(np.int)
        if frame_id - 2 < _other_camera_indices.min():
        # we return here since the other camera info is not availabel
            return []
        # _other_camera_frame = self._othercamera_detection[frame_id - 2]
        _other_mask = (_other_camera_indices == frame_id - 2) & (_other_track_indices != -1)
        _other_rows = self._othercamera_detection[_other_mask]
        if (len(_other_rows) == 0):
            # no available tracklet, return
            return []
        # detections = []
        # detection_indices = []
        # for idx, row in enumerate(_other_rows):
            # bbox, confidence, feature, label = row[2:6], row[6], row[10:], row[7]
            # detections.append(Detection(bbox, confidence, feature, label, idx))
            # detection_indices.append(idx)
        detection_indices = list(range(len(_other_rows)))
        confirmed_tracks = [
            i for i, t in enumerate(self._single_tracker.tracks) if t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(distance_metric, self.metric.matching_threshold, self._single_tracker.tracks, _other_rows,
                confirmed_tracks, detection_indices)
        match_indies = [(self._single_tracker.tracks[track_idx].track_id, _other_rows[detection_idx, 1].astype(np.int)) for track_idx, detection_idx in matches_a]
        return match_indies
            # linear_assignment.min_cost_matching(
            #     gated_metric, self.metric.matching_threshold, self.max_age,
            #     self.tracks, detections, confirmed_tracks)

        # _frame_indices = tracker[:, 0].astype(np.int)
        # _track_indices = tracker[:, 1].astype(np.int)
        # _track_mask = _frame_indices
        # get the current trackid
        # _frame_min = frame_id - 30


        # if _frame_min < _frame_indices.min():
        #     _frame_min = _frame_indices.min()
        #
        # for history_id in range(frame_id, _frame_min, -1):
        #     mask = (_frame_indices == history_id) & (_track_indices != -1)
        #     rows = tracker[mask]
        #
        #     print("current frame {}".format(history_id))

        # now we do cascade matching for each available track id


        # for row in rows:

