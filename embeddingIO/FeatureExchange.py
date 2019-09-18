import os, sys
sys.path.append(os.path.dirname(__file__))

from concurrent import futures
import time
import logging

import grpc

try:
    from embeddings_pb2_grpc import *
    from embeddings_pb2 import *
except:
    from embeddingIO.embeddings_pb2_grpc import *
    from embeddingIO.embeddings_pb2 import *
#
class EmbeddingsServing(EmbServerServicer):
    def __init__(self, trackers, detections):
        self.tracks = trackers
        self.detections = detections
    def getPayload(self, request, context):
        pl = payloads()
        pl.carid = 0
        pl.time = int(time.time())
        total_object = 0
        if request.requestid:
            for track in self.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                # bbox = track.to_tlbr()
                emb = pl.embs.add()
                emb.id = track.track_id
                detection = self.detections[track.detection_id]
                emb.frame_time = detection[0]
                emb.confidence = detection[6]
                emb.labelid = int(detection[7])
                for element in detection[10:]:
                    emb.weight.append(element)
                total_object += 1

        pl.embs_num = total_object
        return pl

