import tensorflow as tf
from detector import *
from detector.PseudoDetector import PseudoDetector
import numpy as np

class TensorFlowDetector(PseudoDetector):

    def __init__(self, sess, checkpoint_filename, class_filter, batch_size=1, altName = None):

        super().__init__(None, altName, False)
        self._batch_size = batch_size
        self._class_filter = class_filter
        self.session = sess
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        model_name = "tensorfow_detector"
        # with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=model_name)
        #     for op in graph.get_operations():
        #         print(op.name)

        # print(tf.global_variables())
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "%s/image_tensor:0" % model_name)

        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = "%s/%s:0" % (model_name, key)
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in self.tensor_dict:
            print("Still does not support segmentation")
            # The following processing is only for single image
            # detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            # detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            # real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            # detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            # detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            # detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            #     detection_masks, detection_boxes, image.shape[0], image.shape[1])
            # detection_masks_reframed = tf.cast(
            #     tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # # Follow the convention by adding back the batch dimension
            # tensor_dict['detection_masks'] = tf.expand_dims(
            #     detection_masks_reframed, 0)

        assert len(self.input_var.get_shape()) == 4
        # self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    # def _encode(self, data_x):
    #     # out = np.zeros((len(data_x), self.feature_dim), np.float32)
    #     run_in_batches(
    #         lambda x: self.session.run(self.output_var, feed_dict=x),
    #         {self.input_var: data_x}, out, self._batch_size)
    #     return out

    def __del__(self):
        try:
            self.session.close()
        except Exception:
            pass

    def __call__(self, img, frameid):
        output_dict = self.session.run(self.tensor_dict, feed_dict={self.input_var: np.expand_dims(img, 0)})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        # for i in range(output_dict['num_detections']):
            # ymin, xmin, ymax, xmax = box
        res = [(frameid, -1, output_dict['detection_boxes'][j][1] * img.shape[1], output_dict['detection_boxes'][j][0] * img.shape[0],
                (output_dict['detection_boxes'][j][3] - output_dict['detection_boxes'][j][1]) * img.shape[1],
                (output_dict['detection_boxes'][j][2] - output_dict['detection_boxes'][j][0]) * img.shape[0],
                output_dict['detection_scores'][j], output_dict['detection_classes'][j], -1, -1) for j in range(output_dict['num_detections'])]
        # detections = [np.r_[raw_detections[id][0:10], features[idx]] for idx, id in enumerate(image_patches_id)]
        # # detections = [Detection(detect_res[id][0], detect_res[id][1], features[idx], detect_res[id][3]) for idx, id in enumerate(image_patches_id)]
        # res = super().__call__(image, detections, frame_id)
        # ymin, xmin, ymax, xmax = box
        if self.isSaveRes:
            self._detections_list.extend(res)
        return res

if __name__ == "__main__":
    # self test function
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.visible_device_list = str(0)
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    import cv2
    img = cv2.imread("/datasets/kitti_tracking/image/0020/000000.png", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frozenpb = r"/home/msis_member/.keras/datasets/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.pb"
    metaPath = r"/home/msis_member/Project/deep_sort/detector/data/kitti_tensorflow.names"
    detector = TensorFlowDetector(sess, frozenpb, class_filter=None, altName=metaPath)

    output = detector(img, 0)

    print(output)

