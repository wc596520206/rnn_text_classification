import tensorflow as tf
from tensorflow.contrib import rnn


class LSTM(object):
    """
    使用LSTM创建模型
    """

    def __init__(self, config):
        """

        :param config: 配置文件
        :param input: 输入的向量
        """
        self.hidden_size = config["model"]["hidden_size"]
        self.dropout = config["model"]["dropout"]
        self.layer_num = config["model"]["layer_num"]
        self.batch_size = config["global"]["batch_size"]

    def _get_a_cell(self, hidden_size, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    def build_net(self, input):
        """
        创建网络
        :param input:
        :return:
        """
        with tf.name_scope("multi_lstm"):
            a = [self._get_a_cell(self.hidden_size, self.dropout) for _ in range(self.layer_num)]  # 多layer_num层
            cell = tf.nn.rnn_cell.MultiRNNCell(a)  # 生成多个
            #initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            out = tf.reduce_mean(outputs, 1)# outputs:64*200*128, 批大小， 时间步， 每个神经元的维度
            return out
