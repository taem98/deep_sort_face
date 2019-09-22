import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np
from deep_sort import linear_assignment
import time
from deep_sort.detection import Detection
from queue import Queue

# import imutils.video

import grpc
# this server should have the ability to create and mantain the server
import os, sys
sys.path.append(os.path.dirname(__file__))

from mctracker import embs_pb2
from mctracker import embs_pb2_grpc

class EmbServer(embs_pb2_grpc.EmbServerServicer):
    def __init__(self, queue):
        self.result_queue = queue
    def sendPayload(self, request, context):
        # this method to handle Payload from client
        try:
            if self.result_queue.full():
                print("Queue full")
                return embs_pb2.respond(respondid=2)
            for idx, emb in enumerate(request.embs):
                weight = [d for d in emb.weight]
                emb = np.asarray([emb.frame_time, emb.id, 0,0,0,0, emb.confidence, emb.labelid, -1, -1, ])
                detection = np.r_[emb, np.asarray(weight)]
                self.result_queue.put(detection)
                # self.result_queue.task_done()
                return embs_pb2.respond(respondid=1)
        except grpc.RpcContext as e:
            print("parsing message error")
            embs_pb2.respond(respondid=0)
        # pass
    def getPayload(self, request, context):
        pass

# def sendPayload(server, detection)

# implement the tracker based on the previous running example from difference camera
# we must change to socket connection later
class MultiCameraTracker:
    def __init__(self, detection_file, metric, single_tracker, detections, bind_addr, server_addr):
        self._single_tracker = single_tracker
        self.detections = detections
        if detection_file is not None:
            self._other_detections = np.load(detection_file)
        else:
            self._other_detections = []
        self.metric = metric

        self.server_addr = server_addr

        self._queue_other_detections = Queue(maxsize=200)
        if len(bind_addr) > 0:
            self.server = grpc.server(ThreadPoolExecutor(max_workers=5))
            self.emb_receiving_sever = EmbServer(self._queue_other_detections)
            embs_pb2_grpc.add_EmbServerServicer_to_server(self.emb_receiving_sever, self.server)
        # embserver = EmbeddingsServing(tracker.tracks, encoder.get_detections())
        # # .add_GreeterServicer_to_server(self.embserver, self.server)
            self.server.add_insecure_port(bind_addr)
            self.server.start()
        # initiate
        if len(self.server_addr) > 0:
            self.client_channel = grpc.insecure_channel(self.server_addr)
            self.client_stub = embs_pb2_grpc.EmbServerStub(self.client_channel)
        # self._mc_detection

    def __del__(self):
        try:
            self.server.stop(2)
        except Exception:
            pass

        try:
            self.client_channel.close()
        except Exception:
            pass

    def broadcastEmb(self):
        '''
        broadcast current tracking result to the air
        actually we send to the omnet simulation or cohda wireless OBU to broadcast via V2X
        '''
        # confirmed_tracks = [
        #     i for i, t in enumerate(self._single_tracker.tracks) if t.is_confirmed()]
        _t0 = time.time()

        payload = embs_pb2.payloads()
        payload.carid = 0 # this id will be mark by simulation tool to ensure uniquely
        payload.time = int(time.time())
        # for idx, t in enumerate(self._single_tracker.tracks1):
        total_object = 0
        for track in self._single_tracker.tracks:
            if not track.is_confirmed():
                continue
            # bbox = track.to_tlbr()
            emb = payload.embs.add()
            emb.id = track.track_id
            detection = self.detections[track.detection_id]
            emb.frame_time = int(detection[0])
            emb.confidence = detection[6]
            emb.labelid = int(detection[7])
            for element in detection[10:]:
                emb.weight.append(element)
            total_object += 1
        payload.embs_num = total_object

        if total_object > 0:
            try:
                respond = self.client_stub.sendPayload(payload)
                print(respond.respondid)
                print("time {}".format(time.time() - _t0))
            except Exception as e:
                print("Send error")


    def agrregate(self, frame_id):

        def distance_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i,10:] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return cost_matrix
        # we get the 2 previous frame from other camera
        _other_camera_indices = self._other_detections[:, 0].astype(np.int)
        _other_track_indices = self._other_detections[:, 1].astype(np.int)
        if frame_id - 2 < _other_camera_indices.min():
        # we return here since the other camera info is not availabel
            return []
        # _other_camera_frame = self._othercamera_detection[frame_id - 2]
        _other_mask = (_other_camera_indices == frame_id - 2) & (_other_track_indices != -1)
        _other_rows = self._other_detections[_other_mask]
        if (len(_other_rows) == 0):
            # no available tracklet, return
            return []
        # finish getting the new space

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

