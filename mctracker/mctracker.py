# this server should have the ability to create and mantain the server
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full
from threading import Thread

import grpc
import numpy as np

from deep_sort import linear_assignment

# import imutils.video
sys.path.append(os.path.dirname(__file__))

from mctracker import embs_pb2
from mctracker import embs_pb2_grpc
from mctracker.mctrack import McTrack

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
# running_mode 0 single track only
# running_mode 1 synchronous : this node is the master to other node
#               2 sync : this node is the slave to other node
#               3 async

class MultiCameraTracker:
    def __init__(self, running_mode, bind_port = 0, server_addr = ""):
        self._single_tracker = None
        self.metric = None
        self.running_mode = running_mode

        if self.running_mode == 0:
            print("MCT Tracker off")
            return

        self.mctracks = []

        self.orphane_tracklets = []

        self._mc_service = Zeroconf()
        self._server_addr = []
        if self.running_mode == 1:
            print("MCT Tracker master mode!!!!")
            if len(server_addr) > 0:
                self._server_addr = server_addr
            else:
                self._server_addr = self._browsing_services()


        _ports, local_addrs = self._register_services(bind_port, "mc_tracker")
        if bind_port == 0:
            bind_port = _ports

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
                        self._server_addr.extend(addr for addr in cmds.master_address)
                        print("Master message {}".format(cmds.cmd))
                        break
                except Exception:
                    time.sleep(0.1)

        self.client_stub = None
        # if len(self._server_addr) > 0:

        cmds = embs_pb2.command()
        cmds.cmd = embs_pb2.command.START
        for addr in local_addrs:
            cmds.master_address.append("{}:{}".format(addr, bind_port))

        for addr in self._server_addr:
            try:
                self.client_channel = grpc.insecure_channel(addr)
                self.client_stub = embs_pb2_grpc.EmbServerStub(self.client_channel)
                # send to test the connection
                respond = self.client_stub.sendCommand(cmds)
                print("Established the connection to {} success".format(addr, respond.respondid))
                break
            except Exception as e:
                # print(e)
                self.client_channel = None
        if self.client_channel is None:
            raise Exception("Cannot connect to server")

        self.client_Q = Queue(maxsize=200)
        self.client_stop = False
        self.client_thread = Thread(target=self._sendEmbs, args=())
        self.client_thread.daemon = True
        self.client_thread.start()

    def updateSingleTracker(self, single_tracker, metric):
        self._single_tracker = single_tracker
        self.metric = metric

    def removeSingleTracker(self):
        self._single_tracker = None
        self.metric = None

        try:
            respond = self.client_stub.sendCommand(embs_pb2.command(cmd=embs_pb2.command.STOPPED))
            respond.respondid = 0
        except Exception:
            pass

    def sendAllPayload(self):
        try:
            respond = self.client_stub.sendCommand(embs_pb2.command(cmd=embs_pb2.command.TRACKLET_SEND))
            respond.respondid = 0
        except Exception:
            pass

    def finished(self):
        if self.running_mode == 1: # this node is the master so we send the cmd to other node
            # while True:
            try:
                respond = self.client_stub.sendCommand(embs_pb2.command(cmd=embs_pb2.command.CONTINUE))
                respond.respondid = 0
            except Exception:
                pass
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

    def initialize_ego_track(self, track):
        if self.running_mode == 0:
            return
        for idx, mctrack in enumerate(self.mctracks):
            if mctrack.ego_track_id == track.track_id:
                mctrack.ego_hit += 1
                mctrack.ego_time_since_update = 0
                mctrack.is_ego_updated = True
                return idx
        self.mctracks.append(McTrack(track.track_id, 3, 60))
        return len(self.mctracks)

    def filter_missing_track(self):
        for mctrack in self.mctracks:
            mctrack.age += 1
            mctrack.remote_time_since_update += 1
            if mctrack.is_ego_updated == True:
                mctrack.ego_time_since_update = 0
                mctrack.is_ego_updated = False
            else:
                mctrack.ego_time_since_update += 1

    def updated_ego_track(self, track):
        for idx, mctrack in enumerate(self.mctracks):
            if mctrack.ego_track_id == track.track_id:
                # mctrack.ego_hit += 1
                return idx
        # self.mctracks.append(McTrack(track.track_id, 2, 60))
        # return len(self.mctracks)

    def agrregate(self, frame_id):
        if self.running_mode == 0:
            return []

        if self._single_tracker is None or self.metric is None:
            print("UPDATE YOUR SINGLE TRACKER AND METRIC FIRST")
            return []

        _detections = []
        # _features = []

        while True:
            try:
                cmds = self._request_Q.get(block=False)
                if cmds.cmd == embs_pb2.command.TRACKLET_SEND:
                    print("Start agrregated {} : {}".format(cmds.cmd, cmds.master_address))

                if cmds.cmd == embs_pb2.command.STOPPED:
                    print("Remote target has close!!!")
                    self.running_mode = 0
                break
            except Exception:
                time.sleep(0.1)


        while not self._payload_Q.empty():
            detection = self._payload_Q.get()
            _detections.append(detection)
            # _features.append(detection[10:])
            self._payload_Q.task_done()
            # time.sleep(0.01)

        _detections = np.asarray(_detections)
        # _features = np.asarray(_features)


        def distance_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i, :] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return cost_matrix
        # we get the 2 previous frame from other camera

        if len(_detections) == 0:
            return []

        detection_indices = list(range(len(_detections)))
        confirmed_tracks = [
            i for i, t in enumerate(self._single_tracker.tracks) if t.is_confirmed()]
        # active_targets.append(self._single_tracker.tracks[track_idx].track_id)
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(distance_metric, 0.3, self._single_tracker.tracks,
                                                _detections[:,10:], confirmed_tracks, detection_indices)
        # match_indies = [(self._single_tracker.tracks[track_idx].track_id,
        #                  _detections[detection_idx, 1].astype(np.int))
        #                 for track_idx, detection_idx in matches_a]
        match_indies = []
        # features, targets, active_targets = [], [], []

        for track_idx, detection_idx in matches_a:
            _index = self.updated_ego_track(self._single_tracker.tracks[track_idx])
            # queue the new id
            self.mctracks[_index].update(_detections[detection_idx,1].astype(np.int), _detections[detection_idx, 10:])
        '''
        the unmatched single tracks and no ego mc tracked can be associate in these situation:
            * the single tracks is recently appear and mc tracked has send in the past
            so we do the comparing function one again
            * 
        '''
        for track_idx in unmatched_tracks_a:
            _index = self.updated_ego_track(self._single_tracker.tracks[track_idx])
            self.mctracks[_index].mark_missed()
        # for any un-associate pair of object, we create the new trackobject without the ego trackid
        for detection_idx in unmatched_detections:
            # may be this node has already here in the remote queue of mctrack, just queue it on
            for idx, mctrack in enumerate(self.mctracks):
                _remote_id = _detections[detection_idx, 1].astype(int)
                if _remote_id in mctrack.remotes_id.keys():
                    self.mctracks[idx].update(_remote_id, _detections[detection_idx, 10:])

            # targets.append()
        # return match_indies

        for idx, mctrack in enumerate(self.mctracks):
            if mctrack.is_confirmed():
                max_id = 0
                max_hits = 0
                # print(type(mctrack.remotes_id))
                for idx, id in enumerate(mctrack.remotes_id.keys()):
                    if max_hits < mctrack.remotes_id[id]:
                        max_id = id
                        max_hits = mctrack.remotes_id[id]
                match_indies.append((mctrack.ego_track_id, max_id))
                for feature in mctrack.features[max_id]:
                    self.metric.samples.setdefault(mctrack.ego_track_id, []).append(feature)
                    if self.metric.budget is not None:
                        self.metric.samples[mctrack.ego_track_id] = self.metric.samples[mctrack.ego_track_id][-self.metric.budget:]
                mctrack.features[max_id] = []

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
                        # fix a wierd zeroconf bug
                        for addr1 in info.properties.values():
                            addresses.append("%s:%d" % (socket.inet_ntoa(addr1), cast(int, info.port)))
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
        # we have some weird bug that address is not fully send so we add to desc
        desc = {str(idx):socket.inet_aton(addr) for idx, addr in enumerate(addrs)}
        info = ServiceInfo(
            "_mctracker._tcp.local.",
            "{}._mctracker._tcp.local.".format(name),
            addresses=[socket.inet_aton(addr) for addr in addrs],
            port=bind_port,
            properties=desc,
        )
        try:
            self._mc_service.register_service(info)
        except Exception as e:
            print("Annce service faild, you have to mannually select the address")
        return bind_port, addrs

    def _sendEmbs(self):
        while True:
            if self.client_stop:
                break
            total_object = 0
            try:
                # _t0 = time.time()
                detection = self.client_Q.get(False)
                payload = embs_pb2.payloads()
                payload.carid = 0  # this id will be mark by simulation tool to ensure uniquely
                payload.time = int(time.time())
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
                respond = self.client_stub.sendPayload(payload)
                # print(respond.respondid)
                # print("time {}".format(time.time() - _t0))
                self.client_Q.task_done()
            except Empty:
                time.sleep(0.01)
            except Exception as e:
                print("Send error {}".format(e))

    def __del__(self):
        try:
            self.server.stop(2)
        except Exception:
            pass

        try:
            self._mc_service.unregister_all_services()
        except Exception:
            pass

        try:
            self.client_stop = True
            self.client_thread.join()
            self.client_channel.close()
        except Exception:
            pass

    def broadcast(self, detection):
        if self.running_mode == 0:
            return
        if self.client_stub is not None:
            self.client_Q.put(detection)




        # for row in rows:

