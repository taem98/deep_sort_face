import os

class kitti_tag(object):
    def __init__(self):
        self.frame = 'frame'
        self.track_id = 'track_id'
        self.label = 'label'
        self.top = 'top'
        self.left = 'left'
        self.bot = 'bot'
        self.right = 'right'
        self.truncated = 'truncated'
        self.occluded = 'occluded'
        self.alpha = 'alpha'
        self.hdim = 'hdim'
        self.wdim = 'wdim'
        self.ldim = 'ldim'
        self.locx, self.locy, self.locz, self.rot_y = "locx", "locy", "locz", "rot_y"
        self.gt_column_tag = [self.frame, self.track_id, self.label, self.truncated,
                                self.occluded, self.alpha, self.left, self.top,
                                self.right, self.bot, self.hdim, self.wdim,
                                self.ldim, self.locx, self.locy, self.locz, self.rot_y]
        self.tag_2d_res = [self.frame, self.track_id, self.label, self.top, self.left, self.bot, self.right]

class Evaluator(object):

    def __init__(self, fmt = "kitti"):
        self.fmt = fmt
        self._results = []
        if self.fmt == "kitti":
            self.tag = kitti_tag()
        self.frameid = -1
        self.altName = None

    def append(self, frameid, trackid, bbox, label=1):
        # to_tlbr self.left, self.top, self.right, self.bot,
        self._results.append([frameid, trackid, label, bbox[1], bbox[0], bbox[3], bbox[2]])

    def append2(self, trackid, bbox, labelid):
        if self.frameid < 0 or not self.altName:
            raise Exception("please set frame id and altName")
        label = self.altName[labelid]
        self.append(self.frameid, trackid, bbox, label)

    def save(self, output_dir, sequence):
        if len(self._results) == 0:
            # append the first DontCare row incase there is no result in the frame
            self._results.append([0, 0, "DontCare", 0, 0, 0, 0])
        kitti_eval_dir = os.path.join(output_dir, "data")
        os.makedirs(kitti_eval_dir, exist_ok=True)
        try:
            import pandas as pd
            df = pd.DataFrame(self._results)
            df.columns = self.tag.tag_2d_res
            if self.fmt == 'kitti':
                df[self.tag.rot_y] = 0
                ignore_tag_1 = [self.tag.truncated, self.tag.occluded, self.tag.alpha, self.tag.hdim, self.tag.wdim,
                                self.tag.ldim]
                for t in ignore_tag_1:
                    df[t] = 0
                ignore_tag_2 = [self.tag.locx, self.tag.locy, self.tag.locz]
                for t in ignore_tag_2:
                    df[t] = -1000
                df.to_csv(os.path.join(kitti_eval_dir, "%s.txt" % sequence), sep=" ", header=False, index=False,
                          columns=self.tag.gt_column_tag)
        except ImportError:
            with open(os.path.join(kitti_eval_dir, "%s.txt" % sequence), 'w') as f:
                for row in self._results:
                     print('%d %d %s 0 0 0 %.2f %.2f %.2f %.2f 0 0 0 -1000 -1000 -1000 0' % (
                        row[0], row[1], row[2], row[3], row[4], row[5], row[6]), file=f)

        self._results = []
        # if not loading_groundtruth and eval_3d is True and(t_data.X==-1000 or t_data.Y==-1000 or t_data.Z==-1000):
        #
        # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
        #     row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
        # KITTI_LABEL = ["frame", "track_id", "class_name", "truncated",
        #                "occluded", "alpha", "bbox_l", "bbox_t",
        #                "bbox_r", "bbox_b", "hdim", "wdim",
        #                "ldim", "locx", "locy", "locz", "rot_y"]