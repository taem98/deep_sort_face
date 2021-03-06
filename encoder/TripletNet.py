import tensorflow as tf
from encoder import *
from encoder.PseudoEncoder import PseudoEncoder

class TripletNet(PseudoEncoder):

    def __init__(self, sess, checkpoint_filename, class_filter, batch_size=32, altName = None):

        super().__init__(None, False, altName)
        self._batch_size = batch_size
        self._class_filter = class_filter
        self.session = sess
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())

        # with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        #     for op in graph.get_operations():
        #         print(op.name)

        # print(tf.global_variables())
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "input:0")
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "head/out_emb:0")

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def _encode(self, data_x):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, self._batch_size)
        return out

    def __del__(self):
        try:
            self.session.close()
        except Exception:
            pass

    def encoding(self, image, raw_detections, frame_id):
        image_patches = []
        image_patches_id = []
        for idx, detect in enumerate(raw_detections):
            if detect[7] in self._class_filter:
                patch = extract_image_patch(image, detect[2:6], self.image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(detect[2:6]))
                    continue
                    # patch = np.random.uniform(
                    #     0., 255., self.image_shape).astype(np.uint8)
                image_patches_id.append(idx)
                image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        features = self._encode(image_patches)
        # return as MOT 16 format with detection
        # detections = [(detect_res[id][0], -1,
        #                detect_res[id][1][0], detect_res[id][1][1], detect_res[id][1][2], detect_res[id][1][3],
        #                detect_res[id][2], detect_res[id][3], -1, -1, features[idx])
        #               for idx, id in enumerate(image_patches_id)]
        detections = [np.r_[raw_detections[id][0:10], features[idx]] for idx, id in enumerate(image_patches_id)]
        return detections

    def __call__(self, image, raw_detections, frame_id):
        detections = self.encoding(image, raw_detections, frame_id)
        # detections = [Detection(detect_res[id][0], detect_res[id][1], features[idx], detect_res[id][3]) for idx, id in enumerate(image_patches_id)]
        res = super().__call__(image, detections, frame_id)
        return res

class FaceNet(PseudoEncoder):

    def __init__(self, sess, checkpoint_filename, class_filter, batch_size=32, altName = None):

        super().__init__(None, False, altName)
        self._batch_size = batch_size
        self._class_filter = class_filter
        self.session = sess

        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())

        # with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="facenet")
        #     for op in graph.get_operations():
        #         print(op.name)

        # print(tf.global_variables())

        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "facenet/input:0")
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "facenet/embeddings:0")

        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("facenet/phase_train:0")

        assert len(self.output_var.get_shape()) == 2
        # assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = [160,160,3]

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def _encode(self, data_x):
        # out = np.zeros((len(data_x), self.feature_dim), np.float32)
        features = []
        for data in data_x:
            prewhiten_face = self.prewhiten(data)
            prewhiten_face = cv2.resize(prewhiten_face, (160, 160))
            feed_dict = {self.input_var: [prewhiten_face], self.phase_train_placeholder: False}
            out = self.session.run(self.output_var, feed_dict= feed_dict)
            features.append(out[0])
        return features


    def __del__(self):
        try:
            self.session.close()
        except Exception:
            pass

    def encoding(self, image, raw_detections, frame_id):
        image_patches = []
        image_patches_id = []
        for idx, detect in enumerate(raw_detections):
            if detect[7] in self._class_filter:
                patch = extract_image_patch(image, detect[2:6], None)
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(detect[2:6]))
                    continue
                    # patch = np.random.uniform(
                    #     0., 255., self.image_shape).astype(np.uint8)
                image_patches_id.append(idx)
                image_patches.append(patch)
        # image_patches = np.asarray(image_patches)
        if len(image_patches):
            features = self._encode(image_patches)
            features = np.asarray(features)
        # return as MOT 16 format with detection
        # detections = [(detect_res[id][0], -1,
        #                detect_res[id][1][0], detect_res[id][1][1], detect_res[id][1][2], detect_res[id][1][3],
        #                detect_res[id][2], detect_res[id][3], -1, -1, features[idx])
        #               for idx, id in enumerate(image_patches_id)]
        detections = [np.r_[raw_detections[id][0:10], features[idx]] for idx, id in enumerate(image_patches_id)]
        return detections

    def __call__(self, image, raw_detections, frame_id):
        detections = self.encoding(image, raw_detections, frame_id)
        # detections = [Detection(detect_res[id][0], detect_res[id][1], features[idx], detect_res[id][3]) for idx, id in enumerate(image_patches_id)]
        res = super().__call__(image, detections, frame_id)
        return res
