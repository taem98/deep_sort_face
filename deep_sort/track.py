# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Predicted = 4
    Registed = 5

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, detection):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        # self.class_labelid = label_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        self.predicted_features = []
        self.is_registed = False
        self.det_meta = []
        self.name = ""

        if detection is not None:
            self.det_meta.append([detection.frame_idx, detection.tlwh, detection.label])
            self.features.append(detection.feature)

        # if label is not None:
        # self.detection_id = detection_id
        self._n_init = n_init
        self._max_age = max_age
        self.affinity_score = 0.0
        self.iou_score = 0.0


    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        if len(self.mean) > 0:
            self.mean, self.covariance = kf.predict(self.mean, self.covariance)
            self.age += 1
            self.time_since_update += 1

    def update_from_predict(self, detection):
        # self.features.append(detection.feature)
        # if self.state == TrackState.Confirmed:
        # previous_label = self.det_meta[-1][2]
        self.det_meta.append([detection[0], self.to_tlwh(), detection[7]])
        # self.state = TrackState.Predicted
        self.hits += 1
        # self.time_since_update = 0
        self.features.append(detection[10:])
        # if self.state == TrackState.Predicted:
        #  should not use the predicted result in consecutive frame

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        if not self.state == TrackState.Registed:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, detection.to_xyah())

        if detection:
            self.det_meta.append([detection.frame_idx, detection.tlwh, detection.label])
            self.features.append(detection.feature)

        # for pf in self.predicted_features:
        #     self.features.append(pf)
        # self.predicted_features = []
        # self.detection_id = detection.detection_id
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Registed and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Predicted:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            if self.is_registed:
                self.state = TrackState.Registed
            else:
                self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def is_predicted(self):
        return  self.state == TrackState.Predicted

    def is_registed_f(self):
        return self.state == TrackState.Registed

    def set_registed_status(self):
        self.state = TrackState.Registed
        self.is_registed = True

    def set_name_feature(self, name, features):
        self.name = name
        self.features = features