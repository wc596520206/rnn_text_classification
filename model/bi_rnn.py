import tensorflow as tf
from tensorflow.contrib import rnn


class BiRNN(object):
    """
    使用RNN创建模型
    """

    def __init__(self, config):
        """

        :param config: 配置文件
        :param input: 输入的向量
        """
        self.hidden_size = config["model"]["hidden_size"]
        self.dropout = config["model"]["dropout"]


    def build_net(self, input):
        rnn_fw_cell = rnn.BasicRNNCell(self.hidden_size)  # forward direction cell
        rnn_bw_cell = rnn.BasicRNNCell(self.hidden_size)  # backward direction cell
        if self.dropout is not None:
            rnn_fw_cell = rnn.DropoutWrapper(
                rnn_fw_cell, output_keep_prob=1 - self.dropout
            )
            rnn_bw_cell = rnn.DropoutWrapper(
                rnn_bw_cell, output_keep_prob=1 - self.dropout
            )
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            rnn_fw_cell, rnn_bw_cell, input, dtype=tf.float32
        )
        bi_output = tf.concat(outputs, axis=2)
        out = tf.reduce_mean(bi_output, 1)
        return out



