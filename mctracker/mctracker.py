# this server should have the ability to create and mantain the server
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import grpc
import numpy as np

from deep_sort import linear_assignment

# import imutils.video
sys.path.append(os.path.dirname(__file__))

from mctracker import embs_pb2
from mctracker import embs_pb2_grpc

from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange

class EmbServer(embs_pb2_grpc.EmbServerServicer):
    def __init__(self, payload_q, request_q):
        self.result_queue = payload_q
        self.request_Q = request_q
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
    def sendCommand(self, request, context):
        self.request_Q.put(request)
        return embs_pb2.respond(respondid=0)
        pass

# def sendPayload(server, detection)

# implement the tracker based on the previous running example from difference camera
# we must change to socket connection later
# running_mode 0 asyncronos
# running_mode 1 synchronous : this node is the master to other node
#               2 sync : this node is the slave to other node


class MultiCameraTracker:
    def __init__(self, detection_file, metric, single_tracker, detections, bind_port, server_addr, running_mode = 0):
        self._single_tracker = single_tracker
        self.detections = detections
        if detection_file is not None:
            self._other_detections = np.load(detection_file)
        else:
            self._other_detections = []
        self.metric = metric
        self.running_mode = running_mode

        self._mc_service = Zeroconf()
        self._server_addr = []
        if self.running_mode == 1:
            print("MCT Tracker master mode!!!!")
            self._server_addr = self._browsing_services()

        # if bind_port == 0:
        bind_port = self._register_services(bind_port, "mc_tracker")

        if bind_port > 0:
            self._request_Q = Queue(maxsize=20)
            self._payload_Q = Queue(maxsize=200)
            self.server = grpc.server(ThreadPoolExecutor(max_workers=5))
            self.emb_receiving_sever = EmbServer(self._payload_Q, self._request_Q)
            embs_pb2_grpc.add_EmbServerServicer_to_server(self.emb_receiving_sever, self.server)
            bind_addr = "[::]:{}".format(int(bind_port))
            self.server.add_insecure_port(bind_addr)
            self.server.start()
            print("Start the MCMT server at {}".format(bind_addr))

        if self.running_mode == 2:
            print("MCMT Tracker slave mode!!!!")
            # wait here until master send the START signal
            while True:
                try:
                    cmds = self._request_Q.get(block=False)
                    if cmds.cmd == embs_pb2.command.START:
                        self._server_addr.append(cmds.master_address)
                        print("Master message {} / {}".format(cmds.cmd, cmds.master_address))
                        break
                except Exception:
                    time.sleep(0.1)

        self.client_stub = None
        # if len(self._server_addr) > 0:
        for addr in self._server_addr:
            try:
                self.client_channel = grpc.insecure_channel(addr)
                print("Connect to {} success".format(addr))
                break
            except Exception as e:
                print(e)
        if self.client_channel is None:
            raise Exception("Cannot connect to slave server")
        self.client_Q = Queue(maxsize=200)
        self.client_stub = embs_pb2_grpc.EmbServerStub(self.client_channel)
        self.client_stop = False

        self.client_thread = Thread(target=self._sendEmbs, args=())
        self.client_thread.daemon = True
        self.client_thread.start()

        if self.running_mode == 1:
            # in master mode we need to tell the slave the current master bind_addess
            try:
                cmds = embs_pb2.command()
                cmds.cmd = embs_pb2.command.START
                cmds.master_address = "localhost:%d" % bind_port
                respond = self.client_stub.sendCommand(cmds)
                if respond.respondid == 0:
                    return
            except Exception as e:
                print(e)
                raise Exception("Error when sending signal to slave node")

        # self._mc_detection

    def _browsing_services(self,):
        from typing import cast
        import socket
        addresses = []
        try:
            def on_service_state_change(
                zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
            ) -> None:
                print("Service %s of type %s state changed: %s" % (name, service_type, state_change))
                if state_change is ServiceStateChange.Added:
                    info = zeroconf.get_service_info(service_type, name)
                    if info:
                        addresses.extend(["%s:%d" % (socket.inet_ntoa(addr), cast(int, info.port)) for addr in info.addresses])
                        print("  Addresses: %s" % ", ".join(addresses))

            browser = ServiceBrowser(self._mc_service, "_mctracker._tcp.local.", handlers=[on_service_state_change])

            try:
                while len(addresses) == 0:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                browser.cancel()

        except Exception as e:
            print("Browsing service failed, mannually input!!!")

        return addresses

    def _register_services(self, bind_port, name=""):
        from mctracker import find_local_address, find_free_port
        import socket
        if bind_port == 0:
            bind_port = find_free_port()[1]

        addrs = find_local_address()
        desc = {'version': '0.10', 'a': 'test value', 'b': 'another value'}
        try:
            info = ServiceInfo(
                "_mctracker._tcp.local.",
                "{}._mctracker._tcp.local.".format(name),
                addresses=[socket.inet_aton(addr) for addr in addrs],
                port=bind_port,
                properties=desc,
            )
            self._mc_service.register_service(info)
        except Exception as e:
            print("Annce service faild, you have to mannually select the address")
        return bind_port

    def _sendEmbs(self):
        while True:
            if self.client_stop:
                break
            total_object = 0
            if not self.client_Q.empty():
                _t0 = time.time()
                detection = self.client_Q.get()
                payload = embs_pb2.payloads()
                payload.carid = 0  # this id will be mark by simulation tool to ensure uniquely
                payload.time = int(time.time())
                # for idx, t in enumerate(self._single_tracker.tracks1):
                # bbox = track.to_tlbr()
                emb = payload.embs.add()
                # detection = self.detections[track.detection_id]
                emb.frame_time = int(detection[0])
                emb.id = int(detection[1])
                emb.confidence = detection[6]
                emb.labelid = int(detection[7])
                for element in detection[10:]:
                    emb.weight.append(element)
                total_object += 1
                payload.embs_num = total_object
                try:
                    respond = self.client_stub.sendPayload(payload)
                    # print(respond.respondid)
                    # print("time {}".format(time.time() - _t0))
                    self.client_Q.task_done()
                except Exception as e:
                    print("Send error {}".format(e))
            else:
                time.sleep(0.1)

    def __del__(self):
        try:
            self.server.stop(2)
        except Exception:
            pass

        try:
            self._mc_service.unregister_all_services()
        except Exception:
            pass

        self.client_stop = True
        self.client_thread.join()

        try:
            self.client_channel.close()
        except Exception:
            pass

    def broadcast(self, detection):
        if self.client_stub is not None:
            self.client_Q.put(detection)

    def finished(self):
        if self.running_mode == 1: # this node is the master so we send the cmd to other node
            # while True:
            respond = self.client_stub.sendCommand(embs_pb2.command(cmd=embs_pb2.command.CONTINUE))
            respond.respondid = 0
        elif self.running_mode == 2: # this node is the slave to other node so wait here until cmd from master arrive
            while True:
                try:
                    cmds = self._request_Q.get(block=False)
                    if cmds.cmd == embs_pb2.command.CONTINUE:
                        print("Frame continue".format(cmds.cmd, cmds.master_address))
                        return
                except Exception:
                    time.sleep(0.1)

        # otherwise just ignore

    def agrregate(self, frame_id):

        def distance_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i, 10:] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return cost_matrix
        # we get the 2 previous frame from other camera
        _other_camera_indices = self._other_detections[:, 0].astype(np.int)
        _other_track_indices = self._other_detections[:, 1].astype(np.int)
        if frame_id  < _other_camera_indices.min():
        # we return here since the other camera info is not availabel
            return []
        # _other_camera_frame = self._othercamera_detection[frame_id - 2]
        _other_mask = (_other_camera_indices == frame_id) & (_other_track_indices != -1)
        _other_rows = self._other_detections[_other_mask]
        if (len(_other_rows) == 0):
            # no available tracklet, return
            return []
        # finish getting the new space

        # if

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
        match_indies = [(self._single_tracker.tracks[track_idx].track_id,
                         _other_rows[detection_idx, 1].astype(np.int))
                        for track_idx, detection_idx in matches_a]
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

