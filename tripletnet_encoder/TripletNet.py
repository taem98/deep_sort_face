import tensorflow as tf
from tripletnet_encoder import *

class TripletNet(object):

    def __init__(self, checkpoint_filename, gpu_num = 0):
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        config.gpu_options.visible_device_list = str(gpu_num)
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

        self.session = tf.Session(config=config)
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

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

    def __del__(self):
        self.session.close()