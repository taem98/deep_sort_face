import os
import time
from visualizer.PoolLoader import PoolLoader, ThreadState
import numpy as np
import cv2

class VideoSplit(PoolLoader):

    def read(self):
        '''
                Read the image in queue and also put this image to display queue
                :return:
                '''

        while True:
            try:
                img = self._reading_queue.get_nowait()
                self.frame_index += 1
                if self.frame_index % self.seq_info["split_ratio"] == 0:
                    break
            except Exception:
                if self._loadingState == ThreadState.Finished:  # this mean loading thread has already finish loading the whole sequence
                    raise Exception("Finished")
                else:
                    time.sleep(0.01)
        while True:
            try:
                self._display_queue.put_nowait(img)
                break
            except Exception:
                time.sleep(0.01)

        # if self._is_not_loading:
        self._reading_queue.task_done()

        print("\r {}/{} Process FPS {} Load {} Playback {} \t\t".format(self.frame_index - 1, self.max_frame_idx,
                                                                        1 / self.last_proctime,
                                                                        self._reading_queue.qsize(),
                                                                        self._display_queue.qsize(),
                                                                        ), end='')
        return self.frame_index - 1, img

    def consume_frame(self):
        while True:
            try:
                img = self._reading_queue.get_nowait()
                self.frame_index += 1
                if self.frame_index % self.seq_info["split_ratio"] == 0:
                    break
            except Exception:
                if self._loadingState == ThreadState.Finished:  # this mean loading thread has already finish loading the whole sequence
                    raise Exception("Finished")
                else:
                    time.sleep(0.01)

        self._reading_queue.task_done()

        print("\r {}/{} Consume frame FPS {} Load {} Playback {} \t\t".format(self.frame_index - 1, self.max_frame_idx,
                                                                        1 / self.last_proctime,
                                                                        self._reading_queue.qsize(),
                                                                        self._display_queue.qsize(),
                                                                        ), end='')
        return self.frame_index - 1, img



def split_video_to_image(metadata, ref_meta, gps_sample, timelist, video_dir, resdir, isdisplay=True):
    seq_info = {}
    seq_info["input_type"] = "video"
    # seq_info["video_path"] = "/datasets/sample_video/2020_01_09/bwm_x5/left/Normal/2020_0109_135128_NOR_FILE.AVI"
    seq_info["update_ms"] = 100
    seq_info["show_detections"] = False
    seq_info["show_tracklets"] = False
    seq_info["split_ratio"] = 3
    pool_display = VideoSplit(20)

    if isdisplay:
        load_func = pool_display.read
    else:
        load_func = pool_display.consume_frame
    # video_metadata_path = "kia_niro_left_metadata.csv"
    gps_samples = np.loadtxt(gps_sample, delimiter=',')
    select_time = np.load(timelist)
    round_time = gps_samples[:, 0]
    selection_mask = np.in1d(round_time, select_time)

    gps_samples = gps_samples[selection_mask]
    # video_dir = '/datasets/sample_video/2020_01_09/kia_niro/left'
    # result_dir = '/datasets/sample_video/2020_01_09/kia_niro/image_left'
    sequence_name = os.path.splitext(os.path.basename(timelist))[0]
    result_gps_sample = os.path.join(resdir, 'gps', sequence_name)
    result_images_dir = os.path.join(resdir, 'images')
    os.makedirs(result_gps_sample, exist_ok=True)
    os.makedirs(result_images_dir, exist_ok=True)
    if ref_meta:
        ref_time = np.loadtxt(ref_meta, delimiter=',', dtype=object)
        ref_time = ref_time[:,0].astype(np.int64)
    gps_samples_index = 0
    frame_idx = 0
    with open(metadata, 'r') as fmeta:
        for idx, line in enumerate(fmeta.readlines()):
            start_time, duration, rel_path = line.rstrip().split(',')
            seq_info["video_path"] = os.path.join(video_dir, rel_path)
            pool_display.load(seq_info)
            if ref_meta:
                try:
                    ref_time_now = ref_time[idx]
                    start_frame = int(start_time)
                    print("time diff {}".format(start_frame - ref_time_now))
                    start_frame = ref_time_now
                except Exception:
                    break
            start_frame = start_frame * 10
            if frame_idx != start_frame and frame_idx != 0:
                print("\n Warning frame mismatch {} to {}".format(frame_idx, start_frame))
            frame_idx = start_frame
            gps_milis_sample = gps_samples[gps_samples_index, 0].astype(np.int64)
            gps_recording_index = gps_samples_index  # index record of first frame in the sequence
            gps_recording_start_frame = gps_milis_sample
            while True:
                try:
                    if gps_samples_index >= gps_samples.shape[0]:
                        break

                    while frame_idx > gps_milis_sample:
                        gps_samples_index += 1
                        gps_milis_sample = gps_samples[gps_samples_index, 0].astype(np.int64)

                    if frame_idx == gps_milis_sample:
                        _, frame = load_func()
                        img_path = os.path.join(result_images_dir, "{}.jpg".format(frame_idx))
                        # if not os.path.isfile(img_path):
                        cv2.imwrite(img_path, frame)

                        gps_samples_index += 1
                        next_sample = gps_samples[gps_samples_index, 0].astype(np.int64)
                        # if the next sample is split after 1 sec
                        if next_sample > gps_milis_sample + 10:
                            output_filename = os.path.join(result_gps_sample,
                                                           "{}.npy".format(gps_recording_start_frame))
                            output_txtname = os.path.join(result_gps_sample, "{}.txt".format(gps_recording_start_frame))

                            np.save(output_filename, gps_samples[gps_recording_index:gps_samples_index],
                                    allow_pickle=False)
                            np.savetxt(output_txtname, gps_samples[gps_recording_index:gps_samples_index], fmt='%4.2f')
                            gps_recording_index = gps_samples_index
                            gps_recording_start_frame = next_sample
                        gps_milis_sample = next_sample
                    else:
                        pool_display.consume_frame()
                    frame_idx += 1
                # except
                except Exception as e:
                    break

    pool_display.stop()

def analyse_result():
    import datetime
    from pytz import timezone
    krt = timezone('Asia/Seoul')
    kia_niro = '/datasets/sample_video/2020_01_09/kia_niro/image_left'
    kia_niro_gps = os.path.join(kia_niro, 'gps', 'round_time_below_25m')
    bmw_x5 = '/datasets/sample_video/2020_01_09/bwm_x5/image_left'
    bmw_x5_gps = os.path.join(bmw_x5, 'gps', 'round_time_below_25m')

    bmw_x5_gps_list = [int(os.path.splitext(f)[0]) for f in os.listdir(bmw_x5_gps) if os.path.splitext(f)[1] == ".npy"]
    niro_gps_list = [int(os.path.splitext(f)[0]) for f in os.listdir(kia_niro_gps) if os.path.splitext(f)[1] == ".npy"]

    bmw_x5_gps_list = np.asarray(bmw_x5_gps_list)
    niro_gps_list = np.asarray(niro_gps_list)

    common_seq = np.intersect1d(bmw_x5_gps_list, niro_gps_list)
    np.save("common_round_time_below_25m.npy", common_seq, allow_pickle=False)
    # print(common_seq)
    for i in common_seq:
        seqname = "{}.npy".format(i)
        bmw_x5_gps_sample_path = os.path.join(bmw_x5_gps, seqname)
        kia_niro_gps_sample_path = os.path.join(kia_niro_gps, seqname)
        bmw_x5_gps_sample = np.load(bmw_x5_gps_sample_path)
        kia_niro_gps_sample = np.load(kia_niro_gps_sample_path)
        if bmw_x5_gps_sample.shape[0] < 60:
            continue
        current_time = datetime.datetime.fromtimestamp(i / 10.0, krt)
        print("{} id {} : bmw {} kia_niro {}".format(current_time, i, bmw_x5_gps_sample.shape[0], kia_niro_gps_sample.shape[0]))
    #     print(bmw_x5_gps_sample_path)
    #     print(kia_niro_gps_sample_path)

def analysis_timestamp():

    kia_gps_samples = np.loadtxt("kia_niro.csv", delimiter=',')
    timestamp = kia_gps_samples[:,0].astype(np.int)
    start = timestamp[0]
    count = 0
    for idx, ts in enumerate(timestamp):
        gps_time_precision = ts - start
        if gps_time_precision == 2:
            print("Frame discontinue!!! {} {}".format(ts, start))
            count += 1
        start = ts
    print(count)
    print("Finish")

def analysis_commontimestamp():

    kia_gps_samples = np.load("round_time_below_40m.npy")
    # timestamp = kia_gps_samples[:,0].astype(np.int)
    start = kia_gps_samples[0]
    count = 0
    for idx, ts in enumerate(kia_gps_samples):
        gps_time_precision = ts - start
        if gps_time_precision > 1:
            print("Frame discontinue!!! {} {}".format(ts, start))
            count += 1
        start = ts
    print(count)
    print("Finish")

if __name__ == '__main__':
    bmw_x5_video_metadata = "bwm_x5_left_metadata.csv"
    kia_niro_metadata = "kia_niro_left_metadata.csv"
    # gps_samples = "bmw_x5.csv"
    # select_time = "round_time_below_25m.npy"
    # video_dir = '/datasets/sample_video/2020_01_09/bwm_x5/left'
    # result_dir = '/datasets/sample_video/2020_01_09/bwm_x5/image_left'
    # split_video_to_image(bmw_x5_video_metadata, gps_samples, select_time, video_dir, result_dir, False)

    # for niro
    # video_metadata_path = "kia_niro_left_metadata.csv"
    gps_samples = "kia_niro.csv"
    select_time = "round_time_below_25m.npy"
    video_dir = '/datasets/sample_video/2020_01_09/kia_niro/left'
    result_dir = '/datasets/sample_video/2020_01_09/kia_niro/image_left'
    # split_video_to_image(kia_niro_metadata, bmw_x5_video_metadata, gps_samples, select_time, video_dir, result_dir, False)

    # split_for_bmw_x5()
    analyse_result()
    # read_video('/datasets/sample_video/2020_01_09/bwm_x5/left/Normal/2020_0109_135128_NOR_FILE.AVI')
    # analysis_timestamp()
    # analysis_commontimestamp()