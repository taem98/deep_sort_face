## python wrapper for detector

from ctypes import *
import math
import random
import os
import cv2
from deep_sort.detection import Detection
from detector.PseudoDetector import PseudoDetector

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/detector/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True

class Detector(PseudoDetector):
    def __init__(self, configPath, weightPath, metaPath, sharelibPath, gpu_num=0):
        """Generate detections results from image.

            Parameters
            ----------
            configPath : path to the cfg of yolo model
            weightPath : path to the weght of selected yolo model
            metaPath : path to the meta of selected yolo model
            sharelibPath : path to the shared library of yolo ( compile detector with USELIB=1
            gpu_num : define which gpu detector will run on
            Returns
            -------
                List[([x,y,w,h],prob, class_id, class_name)]
                Returns detection responses at given frame index.
        """
        super().__init__(False, None, None)
        self._from_file = False
        if os.name == "nt":
            raise Exception("Windows is not support")
        else:
            self.lib = CDLL(sharelibPath, RTLD_GLOBAL)
        self.nw_width = self.lib.network_width
        self.nw_width.argtypes = [c_void_p]
        self.nw_width.restype = c_int
        self.nw_height = self.lib.network_height

        self.nw_height.argtypes = [c_void_p]
        self.nw_height.restype = c_int

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        predict = self.lib.network_predict_ptr
        predict.argtypes = [c_void_p, POINTER(c_float)]
        predict.restype = POINTER(c_float)

        if hasGPU:
            self.set_gpu = self.lib.cuda_set_device
            self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int),
                                      c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        make_network_boxes = self.lib.make_network_boxes
        make_network_boxes.argtypes = [c_void_p]
        make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        free_ptrs = self.lib.free_ptrs
        free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        network_predict = self.lib.network_predict_ptr
        network_predict.argtypes = [c_void_p, POINTER(c_float)]

        reset_rnn = self.lib.reset_rnn
        reset_rnn.argtypes = [c_void_p]

        load_net = self.lib.load_network
        load_net.argtypes = [c_char_p, c_char_p, c_int]
        load_net.restype = c_void_p

        load_net_custom = self.lib.load_network_custom
        load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        load_net_custom.restype = c_void_p

        do_nms_obj = self.lib.do_nms_obj
        do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        letterbox_image = self.lib.letterbox_image
        letterbox_image.argtypes = [IMAGE, c_int, c_int]
        letterbox_image.restype = IMAGE

        load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        rgbgr_image = self.lib.rgbgr_image
        rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        # self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        # self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        # self.predict_image_letterbox.restype = POINTER(c_float)
        self.altNames = None
        self.set_gpu(gpu_num)
        self.netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        self.metaMain = load_meta(metaPath.encode("ascii"))


        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        # self.darknet_image = self.make_image(self.nw_width(self.netMain),
        #                                         self.nw_height(self.netMain), 3)

    def __del__(self):
        pass
        # self.free_image(self.darknet_image)

    def detect_image(self, im, frameid, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        # import cv2
        # custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
        # custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        # custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
        # import scipy.misc
        # custom_image = scipy.misc.imread(image)
        # im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
        num = c_int(0)
        if debug: print("Assigned num")
        pnum = pointer(num)
        if debug: print("Assigned pnum")
        self.predict_image(self.netMain, im)
        letter_box = 0
        # predict_image_letterbox(net, im)
        # letter_box = 1
        if debug: print("did prediction")
        # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
        dets = self.get_network_boxes(self.netMain, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, self.metaMain.classes, nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        class_range = range(self.metaMain.classes)
        if self.altNames is None:
            res = [(frameid, self.metaMain.names[i], dets[j].prob[i], (dets[j].bbox.x, dets[j].bbox.y, dets[j].bbox.w, dets[j].bbox.h))
                   for j in range(num) for i in class_range if dets[j].prob[i] > 0]
        else:
            res = [(frameid, self.altNames[i], dets[j].prob[i], (dets[j].bbox.x, dets[j].bbox.y, dets[j].bbox.w, dets[j].bbox.h))
                   for j in range(num) for i in class_range if dets[j].prob[i] > 0]

        # we try to optimize the speed here
        # for j in range(num):
        #     if debug: print("Ranging on " + str(j) + " of " + str(num))
        #     if debug: print("Classes: " + str(self.metaMain), self.metaMain.classes, self.metaMain.names)
        #     for i in range(self.metaMain.classes):
        #         if debug: print("Class-ranging on " + str(i) + " of " + str(self.metaMain.classes) + "= " + str(dets[j].prob[i]))
        #         if dets[j].prob[i] > 0:
        #             b = dets[j].bbox
        #             if self.altNames is None:
        #                 nameTag = self.metaMain.names[i]
        #             else:
        #                 nameTag = self.altNames[i]
        #             if debug:
        #                 print("Got bbox", b)
        #                 print(nameTag)
        #                 print(dets[j].prob[i])
        #                 print((b.x, b.y, b.w, b.h))
        #             res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        self.free_detections(dets, num)
        if debug: print("freed detections")
        return res

    def __call__(self, img, frameid, thresh=.5, hier_thresh=.5, nms=.45):

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(img,
        #                            (self.nw_width(self.netMain),
        #                             self.nw_height(self.netMain)),
        #                            interpolation=cv2.INTER_LINEAR)
        #
        darknet_image = self.make_image(img.shape[1], img.shape[0], img.shape[2])

        self.copy_image_from_bytes(darknet_image, img.tobytes())
        #
        num = c_int(0)
        # if debug: print("Assigned num")
        pnum = pointer(num)
        # if debug: print("Assigned pnum")
        self.predict_image(self.netMain, darknet_image)
        letter_box = 0
        # predict_image_letterbox(net, im)
        # letter_box = 1
        # if debug: print("did prediction")
        # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
        dets = self.get_network_boxes(self.netMain, darknet_image.w, darknet_image.h, thresh, hier_thresh, None, 0, pnum, letter_box)
        # if debug: print("Got dets")
        num = pnum[0]
        # if debug: print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, self.metaMain.classes, nms)
        # if debug: print("did sort")
        # res = []
        # if debug: print("about to range")
        class_range = range(self.metaMain.classes)
        # res format: [([x,y,w,h],prob, class_id, class_name)]
        # if self.altNames is None:
        #     res = [(frameid, -1, dets[j].bbox.x - dets[j].bbox.w / 2, dets[j].bbox.y - dets[j].bbox.h / 2, dets[j].bbox.w, dets[j].bbox.h,
        #                      dets[j].prob[i], i, -1, -1, self.metaMain.names[i])
        #            for j in range(num) for i in class_range if dets[j].prob[i] > 0]
        # else:
        #     res = [(frameid, -1, dets[j].bbox.x - dets[j].bbox.w / 2, dets[j].bbox.y - dets[j].bbox.h / 2, dets[j].bbox.w, dets[j].bbox.h,
        #                      dets[j].prob[i], i, -1, -1, self.altNames[i])
        #            for j in range(num) for i in class_range if dets[j].prob[i] > 0]
        # ignore the class name, only take class ID
        res = [(frameid, -1, dets[j].bbox.x - dets[j].bbox.w / 2, dets[j].bbox.y - dets[j].bbox.h / 2, dets[j].bbox.w,
                dets[j].bbox.h, dets[j].prob[i], i, -1, -1) for j in range(num) for i in class_range if dets[j].prob[i] > 0]

        self.free_detections(dets, num)
        self.free_image(darknet_image)
        self._detections_list.extend(res)
        return res
